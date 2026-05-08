import argparse
import asyncio
import json
import os
import time
import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from configs.movie.configs import CONFIGS as MOVIE_CONFIGS
from configs.movie.tools import generate_tools as movie_generate_tools
from configs.parallelqa.configs import CONFIGS as PARALLELQA_CONFIGS
from configs.parallelqa.tools import generate_tools as parallelqa_generate_tools
from configs.bfcl_ws.configs import CONFIGS as BFCL_WS_CONFIGS
from configs.bfcl_ws.tools import generate_tools as bfcl_ws_generate_tools
from configs.gaia.configs import CONFIGS as GAIA_CONFIGS
from configs.gaia.tools import generate_tools as gaia_generate_tools

from src.dynacall.controller import Controller
from src.dynacall.scheduler import Scheduler, ResourceBudget
from src.utils.evaluation_utils import compare_answer
from src.utils.logger_utils import enable_logging

from src.utils.embedding_utils import EmbeddingManager
from src.dynacall.llm_adapters import create_llm_adapter

argparser = argparse.ArgumentParser(description="DynaCall with Pre-fetched OpenAI Embeddings")
argparser.add_argument("--N", type=int, default=None, help="number of samples to process")
argparser.add_argument(
    "--row_number",
    type=str,
    default=None,
    help="1-based row selector: single row (3), comma list (1,2,5), or range (1-10)",
)
argparser.add_argument("--logging", action="store_true", help="enable logging")
argparser.add_argument(
    "--model_type",
    type=str,
    default="openai",
    choices=["openai", "vllm", "azure", "friendli"],
    help="model type",
)
argparser.add_argument(
    "--model_name", type=str, default=None, help="model name to override default"
)
argparser.add_argument(
    "--benchmark_name",
    type=str,
    required=True,
    help="benchmark name",
    choices=["movie", "parallelqa", "bfcl_ws", "gaia"],
)
argparser.add_argument("--store", type=str, required=True, help="store path for results")
argparser.add_argument("--do_benchmark", action="store_true", help="do benchmark")
argparser.add_argument(
    "--max_replans",
    type=int,
    default=None,
    help="override max replans (default uses benchmark config)",
)

argparser.add_argument("--max_questions", type=int, default=1,
                      help="Max questions in processing pool")
argparser.add_argument("--max_concurrent", type=int, default=1000,
                      help="Max concurrent for function calls")
argparser.add_argument("--sleep_per_iter", type=int, default=None,
                      help="Sleep seconds per iter to avoid rate limit")

argparser.add_argument("--knn_enabled", action="store_true",
                      help="Enable KNN for execution time prediction")
argparser.add_argument("--knn_k", type=int, default=5,
                      help="Number of neighbors for KNN prediction")
argparser.add_argument("--knn_weights", type=str, default="distance",
                      choices=["uniform", "distance"],
                      help="Weight function for KNN prediction")
argparser.add_argument("--knn_warmup_ratio", type=float, default=0.2,
                      help="Ratio of questions to use for KNN warmup (0.0-1.0)")
argparser.add_argument("--knn_min_samples", type=int, default=1,
                      help="Minimum samples to train KNN")
argparser.add_argument("--knn_metric", type=str, default="euclidean",
                      choices=["euclidean", "manhattan", "minkowski"],
                      help="Distance metric for KNN")

argparser.add_argument("--use_pca", action="store_true",
                      help="Use PCA to reduce embedding dimensions")
argparser.add_argument("--pca_components", type=int, default=50,
                      help="Number of PCA components to keep")

argparser.add_argument("--embedding_model", type=str, default="text-embedding-3-small",
                      help="OpenAI embedding model to use")
argparser.add_argument("--embedding_batch_size", type=int, default=50,
                      help="Batch size for embedding requests")
argparser.add_argument("--embedding_workers", type=int, default=10,
                      help="Number of workers for parallel embedding fetching")

argparser.add_argument("--use_function_coalescing", action="store_true",
                      help="Use function coalescing optimization in DynaCall")
argparser.add_argument("--cache_file", type=str, default=None, help="cache file path")
argparser.add_argument("--use_early_execution", action="store_true", help="stream plan")
argparser.add_argument(
    "--gaia_dataset_path",
    type=str,
    default=None,
    help="Path to the GAIA dataset file (.json or .jsonl)",
)
argparser.add_argument(
    "--gaia_files_root",
    type=str,
    default=None,
    help="Root directory for GAIA attachment files",
)

# vllm-specific arguments
argparser.add_argument("--vllm_port", type=int, default=None, help="vllm port")

args = argparser.parse_args()
if args.logging:
    enable_logging(True)
else:
    enable_logging(False)

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    if candidate.exists():
        return str(candidate.resolve())
    return str((PROJECT_ROOT / candidate).resolve())

def get_dataset(args):
    def _parse_row_selector(selector: Optional[str], dataset_size: int) -> Optional[List[int]]:
        if selector is None:
            return None

        selected: List[int] = []
        seen: Set[int] = set()

        for raw_part in str(selector).split(","):
            part = raw_part.strip()
            if not part:
                continue

            if "-" in part:
                bounds = [item.strip() for item in part.split("-", 1)]
                if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                    raise ValueError(f"Invalid --row_number range: {part}")
                start = int(bounds[0])
                end = int(bounds[1])
                if start <= 0 or end <= 0:
                    raise ValueError("--row_number values must be positive 1-based indices")
                if end < start:
                    raise ValueError(f"Invalid --row_number range: {part}")
                indices = range(start - 1, end)
            else:
                value = int(part)
                if value <= 0:
                    raise ValueError("--row_number values must be positive 1-based indices")
                indices = [value - 1]

            for row_idx in indices:
                if row_idx >= dataset_size:
                    raise IndexError(
                        f"--row_number {row_idx + 1} is out of range for dataset of size {dataset_size}"
                    )
                if row_idx not in seen:
                    seen.add(row_idx)
                    selected.append(row_idx)

        if not selected:
            raise ValueError("--row_number selector produced no rows")
        return selected

    def _apply_row_filter(dataset):
        if args.row_number is None:
            return dataset
        row_indices = _parse_row_selector(args.row_number, len(dataset))
        return [dataset[row_idx] for row_idx in row_indices]

    def _load_json_or_jsonl(path: str):
        path = resolve_project_path(path) or path
        if path.endswith(".jsonl"):
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        return json.load(open(path, "r", encoding="utf-8"))

    def _resolve_gaia_attachment(item, dataset_path: str, files_root: Optional[str]) -> Optional[str]:
        dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
        candidates = []

        file_path = item.get("file_path")
        file_name = item.get("file_name")

        if file_path:
            if os.path.isabs(file_path):
                candidates.append(file_path)
            else:
                candidates.append(os.path.join(dataset_dir, file_path))
                if files_root:
                    candidates.append(os.path.join(files_root, file_path))

        if file_name:
            candidates.append(os.path.join(dataset_dir, file_name))
            if files_root:
                candidates.append(os.path.join(files_root, file_name))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    def _load_gaia_dataset(dataset_path: str, files_root: Optional[str]):
        raw_dataset = _load_json_or_jsonl(dataset_path)
        dataset = []
        for idx, item in enumerate(raw_dataset):
            question = item.get("Question") or item.get("question") or ""
            answer = item.get("Final answer") or item.get("answer") or ""
            question_id = (
                item.get("task_id")
                or item.get("id")
                or item.get("Task ID")
                or f"gaia_{idx}"
            )
            attachment_path = _resolve_gaia_attachment(item, dataset_path, files_root)

            if attachment_path:
                question = (
                    f"{question}\n\n"
                    f"Attached file: {attachment_path}\n"
                    "If the question depends on the attachment, inspect that file before answering."
                )

            dataset.append(
                {
                    "id": str(question_id),
                    "question": question,
                    "answer": answer,
                    "level": item.get("Level") or item.get("level"),
                    "file_name": item.get("file_name"),
                    "file_path": attachment_path,
                }
            )
        return _apply_row_filter(dataset)

    dataset_name = "datasets/"
    if args.benchmark_name == "movie":
        dataset_name = "datasets/movie_recommendation.json"
    elif args.benchmark_name == "parallelqa":
        dataset_name = "datasets/parallelqa_dataset.json"
    elif args.benchmark_name == "bfcl_ws":
        dataset_name = "datasets/BFCL_v4_web_search_processed.json"
    elif args.benchmark_name == "gaia":
        dataset_path = resolve_project_path(
            args.gaia_dataset_path or "datasets/gaia/gaia_validation.jsonl"
        )
        files_root = resolve_project_path(
            args.gaia_files_root or "datasets/gaia/raw/2023/validation"
        )
        return _load_gaia_dataset(dataset_path, files_root)
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")

    dataset_name = resolve_project_path(dataset_name) or dataset_name
    dataset = json.load(open(dataset_name, "r"))
    return _apply_row_filter(dataset)

def get_tools(model_name, args):
    
    if args.benchmark_name == "movie":
        tools = movie_generate_tools(args)
    elif args.benchmark_name == "parallelqa":
        tools = parallelqa_generate_tools(args, model_name)
    elif args.benchmark_name == "bfcl_ws":
        tools = bfcl_ws_generate_tools(args, model_name)
    elif args.benchmark_name == "gaia":
        tools = gaia_generate_tools(args, model_name)
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
    return tools

def get_configs(args):
    
    if args.benchmark_name == "movie":
        configs = MOVIE_CONFIGS
    elif args.benchmark_name == "parallelqa":
        configs = PARALLELQA_CONFIGS
    elif args.benchmark_name == "bfcl_ws":
        configs = BFCL_WS_CONFIGS
    elif args.benchmark_name == "gaia":
        configs = GAIA_CONFIGS
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
    return configs

class KNNPredictor:
    
    
    def __init__(self, k: int = 5, weights: str = 'distance', 
                 metric: str = 'euclidean', use_pca: bool = False, 
                 pca_components: int = 50):
        self.k = k
        self.weights = weights 
        self.metric = metric
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.embedding_manager = None
        
        
        self.model = KNeighborsRegressor(
            n_neighbors=min(k, 10),  
            weights=weights,
            metric=metric,
            algorithm='auto',
            leaf_size=30,
            p=2,  
            n_jobs=-1  
        )
        
        
        self.scaler = StandardScaler()
        
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        
        self.X_train_ids = []  # question_ids
        self.y_train = []      # execution times
        self.X_train_embeddings = []  
        
        
        self.training_stats = {}
        self.is_trained = False
        
    def set_embedding_manager(self, manager: EmbeddingManager):
        
        self.embedding_manager = manager
    
    def add_training_sample(self, question_id: str, execution_time: float):
        
        self.X_train_ids.append(question_id)
        self.y_train.append(execution_time)
    
    def train(self):
        
        if len(self.X_train_ids) < max(3, self.k):  
            print(f"  Not enough training samples: {len(self.X_train_ids)} < {max(3, self.k)}")
            return False
        
        
        X_embeddings = []
        valid_ids = []
        valid_times = []
        
        for qid, exec_time in zip(self.X_train_ids, self.y_train):
            embedding = self.embedding_manager.get_embedding(qid)
            if embedding is not None:
                X_embeddings.append(embedding)
                valid_ids.append(qid)
                valid_times.append(exec_time)
        
        if len(X_embeddings) < max(3, self.k):
            print(f"  Not enough valid embeddings: {len(X_embeddings)} < {max(3, self.k)}")
            return False
        
        X = np.array(X_embeddings)
        y = np.array(valid_times)
        
        
        self.X_train_ids = valid_ids
        self.y_train = valid_times
        self.X_train_embeddings = X
        
        
        print(f" KNN training data statistics:")
        print(f"   Samples: {len(X)}")
        print(f"   k: {self.k}")
        print(f"   Weights: {self.weights}")
        print(f"   Metric: {self.metric}")
        print(f"   PCA: {'Enabled' if self.use_pca else 'Disabled'}")
        print(f"   Execution times: min={y.min():.1f}s, max={y.max():.1f}s, mean={y.mean():.1f}s, std={y.std():.1f}s")
        
        
        self.training_stats = {
            'n_samples': len(X),
            'k': self.k,
            'weights': self.weights,
            'metric': self.metric,
            'y_min': float(y.min()),
            'y_max': float(y.max()),
            'y_mean': float(y.mean()),
            'y_std': float(y.std()),
            'y_median': float(np.median(y)),
        }
        
        
        actual_k = min(self.k, len(X) - 1)  
        if actual_k != self.k:
            print(f"   Adjusted k from {self.k} to {actual_k} (limited by sample size)")
            self.model.set_params(n_neighbors=actual_k)
        else:
            self.model.set_params(n_neighbors=self.k)
        
        
        if self.use_pca and X.shape[1] > self.pca_components:
            print(f"   Applying PCA: {X.shape[1]} -> {self.pca_components} dimensions")
            X = self.pca.fit_transform(X)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"   Explained variance: {explained_variance:.3f}")
            self.training_stats['pca_explained_variance'] = float(explained_variance)
        
        
        print(f"   Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        
        feature_std = np.std(X_scaled, axis=0)
        zero_variance_features = np.sum(feature_std < 1e-6)
        if zero_variance_features > 0:
            print(f"  Warning: {zero_variance_features} features have near-zero variance")
        
        
        print(f" Training KNN model...")
        try:
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            
            y_pred = self.model.predict(X_scaled)
            
            mae = np.mean(np.abs(y - y_pred))
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            mape = np.mean(np.abs((y - y_pred) / y)) * 100 if np.all(y > 0) else float('inf')
            
            
            self.training_stats.update({
                'training_mae': mae,
                'training_rmse': rmse,
                'training_mape': mape,
                'is_trained': True
            })
            
            print(f" KNN model trained successfully")
            print(f"   Training MAE: {mae:.2f}s, RMSE: {rmse:.2f}s, MAPE: {mape:.1f}%")
            
            
            print(f"\n Sample predictions from training set (first 5):")
            for i in range(min(5, len(y))):
                print(f"   {i+1}. {self.X_train_ids[i]}: Actual={y[i]:.1f}s | Pred={y_pred[i]:.1f}s | Diff={y[i]-y_pred[i]:.1f}s")
            
            return True
            
        except Exception as e:
            print(f" KNN training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_features(self, question_id: str) -> Optional[np.ndarray]:
        
        if not self.embedding_manager:
            return None
        
        embedding = self.embedding_manager.get_embedding(question_id)
        if embedding is None:
            return None
        
        
        embedding = embedding.reshape(1, -1)
        
        
        if self.use_pca and self.pca is not None and hasattr(self.pca, 'components_'):
            try:
                embedding = self.pca.transform(embedding)
            except Exception as e:
                print(f"  PCA transform error: {e}")
        
        return embedding
    
    def predict(self, question_id: str) -> Tuple[float, float]:
        
        if not self.is_trained or not self.embedding_manager:
            print(f"  KNN not trained, returning default for {question_id}")
            if hasattr(self, 'training_stats') and 'y_mean' in self.training_stats:
                return self.training_stats['y_mean'], self.training_stats['y_std']
            return 30.0, 10.0
        
        embedding = self._prepare_features(question_id)
        if embedding is None:
            return self.training_stats['y_mean'], self.training_stats['y_std']
        
        try:
            
            X_scaled = self.scaler.transform(embedding)
            
            
            y_pred = self.model.predict(X_scaled)[0]
            
            
            distances, indices = self.model.kneighbors(X_scaled, return_distance=True)
            
            
            neighbor_times = []
            neighbor_weights = []
            
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.y_train):
                    neighbor_times.append(self.y_train[idx])
                    
                    weight = 1.0 / (dist + 1e-10) if self.weights == 'distance' else 1.0
                    neighbor_weights.append(weight)
            
            if neighbor_times:
                
                weights = np.array(neighbor_weights)
                weights = weights / weights.sum()  
                
                weighted_mean = np.average(neighbor_times, weights=weights)
                weighted_variance = np.average((neighbor_times - weighted_mean) ** 2, weights=weights)
                weighted_std = np.sqrt(weighted_variance)
                
                
                if len(neighbor_times) == 1:
                    weighted_std = self.training_stats['y_std']
            else:
                weighted_std = self.training_stats['y_std']
            
            
            y_pred = max(1.0, y_pred)
            weighted_std = max(0.1, min(weighted_std, 50.0))
            
            return y_pred, weighted_std
            
        except Exception as e:
            print(f"  KNN prediction error for {question_id}: {e}")
            import traceback
            traceback.print_exc()
            return self.training_stats['y_mean'], self.training_stats['y_std']
    
    def predict_all(self, question_ids: List[str]) -> List[Tuple[str, float, float]]:
        
        if not self.is_trained:
            print("  KNN not trained, returning default predictions")
            return [(qid, 30.0, 10.0) for qid in question_ids]
        
        predictions = []
        unique_predictions = set()
        
        print(f" Predicting {len(question_ids)} questions using KNN (k={self.k})...")
        
        
        batch_embeddings = []
        valid_ids = []
        
        for qid in question_ids:
            embedding = self._prepare_features(qid)
            if embedding is not None:
                batch_embeddings.append(embedding[0])  
                valid_ids.append(qid)
        
        if batch_embeddings:
            X = np.array(batch_embeddings)
            
            
            if self.use_pca and self.pca is not None:
                try:
                    X = self.pca.transform(X)
                except:
                    pass
            
            
            X_scaled = self.scaler.transform(X)
            
            
            try:
                y_preds = self.model.predict(X_scaled)
                
                
                for i, (qid, y_pred) in enumerate(zip(valid_ids, y_preds)):
                    
                    x_sample = X_scaled[i:i+1]
                    distances, indices = self.model.kneighbors(x_sample, return_distance=True)
                    
                    
                    neighbor_times = []
                    neighbor_weights = []
                    
                    for idx, dist in zip(indices[0], distances[0]):
                        if idx < len(self.y_train):
                            neighbor_times.append(self.y_train[idx])
                            weight = 1.0 / (dist + 1e-10) if self.weights == 'distance' else 1.0
                            neighbor_weights.append(weight)
                    
                    if neighbor_times:
                        weights = np.array(neighbor_weights)
                        weights = weights / weights.sum()
                        weighted_mean = np.average(neighbor_times, weights=weights)
                        weighted_variance = np.average((neighbor_times - weighted_mean) ** 2, weights=weights)
                        weighted_std = np.sqrt(weighted_variance)
                    else:
                        weighted_std = self.training_stats['y_std']
                    
                    
                    y_pred = max(1.0, y_pred)
                    weighted_std = max(0.1, min(weighted_std, 50.0))
                    
                    predictions.append((qid, y_pred, weighted_std))
                    unique_predictions.add((round(y_pred, 2), round(weighted_std, 2)))
                    
            except Exception as e:
                print(f" Batch prediction failed: {e}, falling back to individual predictions")
                
                for qid in question_ids:
                    pred_time, pred_std = self.predict(qid)
                    predictions.append((qid, pred_time, pred_std))
                    unique_predictions.add((round(pred_time, 2), round(pred_std, 2)))
        else:
            
            for i, qid in enumerate(question_ids):
                pred_time, pred_std = self.predict(qid)
                predictions.append((qid, pred_time, pred_std))
                unique_predictions.add((round(pred_time, 2), round(pred_std, 2)))
                
                
                if (i + 1) % 10 == 0:
                    print(f"   Predicted {i+1}/{len(question_ids)}")
        
        
        if predictions:
            pred_times = [p[1] for p in predictions]
            pred_stds = [p[2] for p in predictions]
            
            print(f"\n KNN prediction statistics:")
            print(f"   Times: min={min(pred_times):.1f}s, max={max(pred_times):.1f}s, mean={np.mean(pred_times):.1f}s")
            print(f"   Stds: min={min(pred_stds):.1f}s, max={max(pred_stds):.1f}s, mean={np.mean(pred_stds):.1f}s")
            print(f"   Unique predictions: {len(unique_predictions)}")
            
            if len(unique_predictions) <= 3:
                print(f"  Warning: Very few unique predictions ({len(unique_predictions)})")
        
        return predictions

class RealTimeResultManager:
    
    
    def __init__(self, store_path: str, labels_map: Dict[str, Any]):
        self.store_path = store_path
        self.labels_map = labels_map
        self.all_results = self._load_existing_results()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.batch_start_time = None
        self.batch_end_time = None
        
    def _load_existing_results(self) -> Dict[str, Any]:
        
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  Failed to load existing results: {e}")
                return {}
        return {}
    
    def set_batch_start_time(self, start_time: float):
        
        self.batch_start_time = start_time
    
    def set_batch_end_time(self, end_time: float):
        
        self.batch_end_time = end_time
    
    def update_result(self, question_id: str, result: Dict[str, Any]):
        
        with self.lock:
            start_time = result.get("start_time")
            end_time = result.get("end_time")
            actual_time = end_time - start_time if start_time and end_time else result.get("time", 0)
            
            complete_result = {
                "question": result["question"],
                "label": self.labels_map.get(question_id, "Unknown"),
                "answer": result["answer"],
                "time": actual_time,
                "status": result["status"],
                "knn_enabled": args.knn_enabled,
                "knn_k": args.knn_k if args.knn_enabled else None,
                "knn_weights": args.knn_weights if args.knn_enabled else None,
                "start_time": start_time,
                "end_time": end_time,
                "start_time_str": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)) if start_time else None,
                "end_time_str": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)) if end_time else None,
                "original_tasks": result.get("original_tasks", 0),
                "optimized_tasks": result.get("optimized_tasks", 0),
                "cached_tasks": result.get("cached_tasks", 0),
                "executed_tasks": result.get("executed_tasks", 0),
                "shared_task_hits": result.get("shared_task_hits", 0),
                "leaf_tasks": result.get("leaf_tasks", 0),
                "stats": result.get("stats"),
            }
            
            self.all_results[question_id] = complete_result
            self._save_async()
    
    def _save_async(self):
        
        def save():
            try:
                temp_path = self.store_path + '.tmp'
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.all_results, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, self.store_path)
            except Exception as e:
                print(f" Failed to save results: {e}")
        
        self.executor.submit(save)

        
    def get_results(self) -> Dict[str, Any]:
        
        with self.lock:
            return self.all_results.copy()
    
    def get_progress(self) -> tuple:
        
        with self.lock:
            total = len(self.all_results)
            successful = len([r for r in self.all_results.values() if r.get("status") == "success"])
            failed = len([r for r in self.all_results.values() if r.get("status") == "error"])
            return total, successful, failed
    
    def get_existing_question_ids(self) -> Set[str]:
        
        with self.lock:
            return set(self.all_results.keys())
    
    def get_batch_time_stats(self) -> Dict[str, Any]:
        
        with self.lock:
            successful_results = [
                r for r in self.all_results.values() 
                if r.get("status") == "success" and r.get("start_time") and r.get("end_time")
            ]
            
            if not successful_results:
                return {
                    "batch_total_time": 0,
                    "earliest_start": 0,
                    "latest_end": 0,
                    "actual_batch_duration": 0,
                    "avg_question_time": 0,
                    "parallel_efficiency": 0,
                    "total_sequential_time": 0,
                    "questions_processed": 0
                }
            
            start_times = []
            end_times = []
            question_times = []
            
            for r in successful_results:
                start_time = r["start_time"]
                end_time = r["end_time"]
                
                if isinstance(start_time, str):
                    try:
                        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp()
                    except:
                        continue
                if isinstance(end_time, str):
                    try:
                        end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').timestamp()
                    except:
                        continue
                
                start_times.append(start_time)
                end_times.append(end_time)
                question_times.append(end_time - start_time)
            
            if not start_times or not end_times:
                return {
                    "batch_total_time": 0,
                    "earliest_start": 0,
                    "latest_end": 0,
                    "actual_batch_duration": 0,
                    "avg_question_time": 0,
                    "parallel_efficiency": 0,
                    "total_sequential_time": 0,
                    "questions_processed": 0
                }
            
            earliest_start = min(start_times)
            latest_end = max(end_times)
            actual_batch_duration = latest_end - earliest_start
            
            avg_question_time = np.mean(question_times) if question_times else 0
            total_sequential_time = sum(question_times)
            parallel_efficiency = total_sequential_time / (actual_batch_duration * len(successful_results)) if actual_batch_duration > 0 else 0
            
            batch_total_time = self.batch_end_time - self.batch_start_time if self.batch_start_time and self.batch_end_time else actual_batch_duration
            
            return {
                "batch_total_time": batch_total_time,
                "earliest_start": earliest_start,
                "latest_end": latest_end,
                "actual_batch_duration": actual_batch_duration,
                "avg_question_time": avg_question_time,
                "total_sequential_time": total_sequential_time,
                "parallel_efficiency": parallel_efficiency,
                "questions_processed": len(successful_results)
            }
    
    def shutdown(self):
        
        self.executor.shutdown(wait=True)

def calculate_final_metrics(all_results: Dict[str, Any], benchmark_name: Optional[str] = None) -> Dict[str, Any]:
    
    successful_results = [
        result for result in all_results.values() 
        if result.get("status") == "success"
    ]
    
    if not successful_results:
        return {
            "accuracy": 0.0,
            "latency_avg": 0.0,
            "latency_std": 0.0,
            "total_count": 0,
            "successful_count": 0,
            "failed_count": 0
        }
    
    accuracy_scores = []
    for result in successful_results:
        answer = result.get("answer", "")
        label = result.get("label", "")
        if type(label) == list:
            score_list = []
            for item in label:
                score = compare_answer(answer, item, benchmark_name=benchmark_name)
                score_list.append(score)
            score = max(score_list)
        else:
            score = compare_answer(answer, label, benchmark_name=benchmark_name)
        accuracy_scores.append(score)
    
    accuracy = np.mean(accuracy_scores)
    
    latencies = [result.get("time", 0) for result in successful_results]
    latency_avg = np.mean(latencies)
    latency_std = np.std(latencies)
    
    task_stats = {}
    if successful_results and "original_tasks" in successful_results[0]:
        original_tasks = sum([r.get("original_tasks", 0) for r in successful_results])
        optimized_tasks = sum([r.get("optimized_tasks", 0) for r in successful_results])
        cached_tasks = sum([r.get("cached_tasks", 0) for r in successful_results])
        executed_tasks = sum([r.get("executed_tasks", 0) for r in successful_results])
        shared_task_hits = sum([r.get("shared_task_hits", 0) for r in successful_results])
        
        optimization_rate = (original_tasks - executed_tasks) / original_tasks if original_tasks > 0 else 0.0
        cache_hit_rate = cached_tasks / (cached_tasks + executed_tasks) if (cached_tasks + executed_tasks) > 0 else 0.0
        
        task_stats = {
            "original_tasks": original_tasks,
            "optimized_tasks": optimized_tasks,
            "cached_tasks": cached_tasks,
            "executed_tasks": executed_tasks,
            "shared_task_hits": shared_task_hits,
            "optimization_rate": optimization_rate,
            "cache_hit_rate": cache_hit_rate
        }

    token_stats = {}
    token_rows = []
    for result in successful_results:
        total_stats = ((result.get("stats") or {}).get("total") or {})
        if total_stats:
            token_rows.append(total_stats)
    if token_rows:
        total_calls = sum(row.get("calls", 0) or 0 for row in token_rows)
        total_input_tokens = sum(row.get("input_tokens", 0) or 0 for row in token_rows)
        total_output_tokens = sum(row.get("output_tokens", 0) or 0 for row in token_rows)
        total_tokens = sum(row.get("total_tokens", 0) or 0 for row in token_rows)
        denom = len(token_rows)
        token_stats = {
            "count": denom,
            "total_calls": total_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "avg_calls": total_calls / denom,
            "avg_input_tokens": total_input_tokens / denom,
            "avg_output_tokens": total_output_tokens / denom,
            "avg_total_tokens": total_tokens / denom,
        }
    
    total_count = len(all_results)
    successful_count = len(successful_results)
    failed_count = total_count - successful_count
    
    return {
        "accuracy": accuracy,
        "latency_avg": latency_avg,
        "latency_std": latency_std,
        "total_count": total_count,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "task_stats": task_stats,
        "token_stats": token_stats,
    }

async def process_with_prefetched_embeddings(agent, dataset, args):
    
    
    questions_to_process = []
    labels_map = {}
    
    for i, example in enumerate(dataset):
        if args.N and i >= args.N:
            break
        qid = str(example["id"])
        question = example["question"]
        _label = example["answer"]
        
        questions_to_process.append({
            'id': qid,
            'question': question
        })
        labels_map[qid] = _label
    
    if not questions_to_process:
        print(" No questions to process!")
        return {}
    
    print(f" Starting processing with pre-fetched embeddings")
    print(f"   Total questions: {len(questions_to_process)}")
    print(f"   KNN enabled: {args.knn_enabled}")
    
    
    
    knn_predictor = None
    if args.knn_enabled:
        
        cache_path = args.store
        if cache_path.endswith('/'): cache_path = cache_path[:-1]
        cache_path = '/'.join(args.store.split('/')[:-1]) + '/{}_embeddings.npy'.format(args.benchmark_name)
        embedding_api_key = os.environ["OPENAI_API_KEY"]
        embedding_manager = EmbeddingManager(api_key=embedding_api_key,
                                            cache_path=cache_path, model=args.embedding_model)
        knn_predictor = KNNPredictor(
            k=args.knn_k,
            weights=args.knn_weights,
            metric=args.knn_metric,
            use_pca=args.use_pca,
            pca_components=args.pca_components
        )
        knn_predictor.set_embedding_manager(embedding_manager)
        print(f"   KNN k: {args.knn_k}")
        print(f"   KNN weights: {args.knn_weights}")
        print(f"   KNN metric: {args.knn_metric}")
        print(f"   KNN warmup ratio: {args.knn_warmup_ratio}")
        print(f"   Minimum samples: {args.knn_min_samples}")
    
    
    result_manager = RealTimeResultManager(args.store, labels_map)
    
    
    existing_question_ids = result_manager.get_existing_question_ids()
    filtered_questions = [
        q for q in questions_to_process 
        if q['id'] not in existing_question_ids
    ]
    
    if not filtered_questions:
        print("All questions have been processed already!")
        all_results = result_manager.get_results()
        final_metrics = calculate_final_metrics(all_results, benchmark_name=args.benchmark_name)
        batch_stats = result_manager.get_batch_time_stats()
        _print_final_metrics(final_metrics, batch_stats, args)
        result_manager.shutdown()
        return all_results
    
    print(f" {len(filtered_questions)} new questions to process")
    
    
    if args.knn_enabled and args.knn_warmup_ratio > 0:
        await embedding_manager.prefetch_embeddings(filtered_questions)

        warmup_size = int(len(filtered_questions) * args.knn_warmup_ratio)
        warmup_size = max(args.knn_min_samples, warmup_size)
        
        if warmup_size >= len(filtered_questions):
            warmup_size = max(1, len(filtered_questions) // 2)
        
        print(f"\n KNN warmup phase: {warmup_size} questions")
        
        
        warmup_questions = filtered_questions[:warmup_size]
        remaining_questions = filtered_questions[warmup_size:]
        
        
        if warmup_questions:
            print(f"   Processing {len(warmup_questions)} warmup questions...")
            warmup_start = time.time()
            
            await _process_questions_batch(
                agent, warmup_questions, result_manager, args
            )
            
            warmup_time = time.time() - warmup_start
            print(f"   Warmup completed in {warmup_time:.2f}s")
            
            
            print(f"   Collecting execution times for KNN training...")
            collected_samples = 0
            for q in warmup_questions:
                qid = q['id']
                if qid in result_manager.all_results:
                    result = result_manager.all_results[qid]
                    if result.get("status") == "success":
                        exec_time = result.get("time", 0)
                        if exec_time > 0:
                            knn_predictor.add_training_sample(qid, exec_time)
                            collected_samples += 1
                            print(f"     {qid}: {exec_time:.1f}s")
            
            print(f"   Collected {collected_samples} training samples")
        
        
        if collected_samples >= args.knn_min_samples:
            if knn_predictor.train():
                print(f" KNN model trained successfully")
                
                if remaining_questions:
                    print(f"\n Predicting execution times for {len(remaining_questions)} questions...")
                    predictions_start = time.time()
                    
                    
                    remaining_ids = [q['id'] for q in remaining_questions]
                    predictions = knn_predictor.predict_all(remaining_ids)
                    
                    
                    question_map = {q['id']: q for q in remaining_questions}
                    
                    
                    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
                    
                    
                    remaining_questions = []
                    for qid, pred_time, pred_std in sorted_predictions:
                        if qid in question_map:
                            remaining_questions.append(question_map[qid])
                        print(qid, pred_time)
                    predictions_time = time.time() - predictions_start
                    
                    
                    if sorted_predictions:
                        times = [p[1] for p in sorted_predictions]
                        print(f"   Predictions: min={min(times):.1f}s, avg={np.mean(times):.1f}s, max={max(times):.1f}s")
                        print(f"   Prediction time: {predictions_time:.2f}s")
                    print(f"   Reordered {len(remaining_questions)} questions by predicted execution time")
            else:
                print(f"  KNN model training failed, using original order")
        
        
        filtered_questions = remaining_questions
    
    
    print(f"\n Main processing phase: {len(filtered_questions)} questions")
    
    
    budget = ResourceBudget(
        max_questions=min(args.max_questions, len(filtered_questions)),
        max_concurrent=args.max_concurrent
    )
    
    
    scheduler = Scheduler(
        agent=agent,
        budget=budget,
        cache_file=args.cache_file,
        result_callback=result_manager.update_result,
        enable_early_execution=args.use_early_execution,
        enable_function_coalescing=args.use_function_coalescing
    )
    
    
    batch_start_time = time.time()
    result_manager.set_batch_start_time(batch_start_time)
    
    try:
        print(f" Starting processing at {time.strftime('%H:%M:%S')}")
        
        
        progress_task = asyncio.create_task(_monitor_progress(
            result_manager, batch_start_time, len(filtered_questions)
        ))
        
        
        results = await scheduler.process_questions(filtered_questions)
        
        batch_end_time = time.time()
        result_manager.set_batch_end_time(batch_end_time)
        
        
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        
        
        result_manager._save_async()
        
        
        all_results = result_manager.get_results()
        final_metrics = calculate_final_metrics(all_results, benchmark_name=args.benchmark_name)
        batch_stats = result_manager.get_batch_time_stats()
        
        
        if args.knn_enabled and knn_predictor and knn_predictor.is_trained:
            _print_knn_statistics(knn_predictor, all_results)
        
        
        total, successful, failed = result_manager.get_progress()
        print(f"\n Processing completed!")
        print(f"  Batch total time: {batch_stats['batch_total_time']:.2f}s")
        
        
        _print_final_metrics(final_metrics, batch_stats, args)
        
        print(f" Results saved to: {args.store}")
        
    except Exception as e:
        print(f" processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        
        result_manager.shutdown()
    
    return result_manager.get_results()

async def _process_questions_batch(agent, questions, result_manager, args):
    
    if not questions:
        return {}
    
    
    budget = ResourceBudget(
        max_questions=min(args.max_questions, len(questions)),
        max_concurrent=args.max_concurrent
    )
    
    
    scheduler = Scheduler(
        agent=agent,
        budget=budget,
        cache_file=args.cache_file,
        result_callback=result_manager.update_result,
        enable_early_execution=args.use_early_execution,
        enable_function_coalescing=args.use_function_coalescing
    )
    
    
    return await scheduler.process_questions(questions)

async def _monitor_progress(result_manager: RealTimeResultManager, start_time: float, total_questions: int):
    
    last_report_time = start_time
    last_count = 0
    
    while True:
        await asyncio.sleep(5)  
        
        current_time = time.time()
        total, successful, failed = result_manager.get_progress()
        completed = successful + failed
        
        
        elapsed = current_time - start_time
        if elapsed > 0:
            speed = completed / elapsed  
            eta = (total_questions - completed) / speed if speed > 0 else 0
        else:
            speed = 0
            eta = 0
        
        print(f"Progress: {completed}/{total_questions} "
              f"({completed/total_questions*100:.1f}%) | "
              f" {successful} |  {failed} | "
              f"Speed: {speed*60:.2f}q/min | ETA: {eta/60:.1f}min")
        
        
        if completed >= total_questions:
            break

def _print_knn_statistics(knn_predictor, all_results):
    
    print("\n" + "="*60)
    print("KNN PREDICTION STATISTICS")
    print("="*60)
    
    if not knn_predictor.is_trained:
        print("KNN model not trained")
        return
    
    
    valid_samples = []
    for qid, exec_time in zip(knn_predictor.X_train_ids, knn_predictor.y_train):
        if qid in all_results:
            result = all_results[qid]
            if result.get("status") == "success":
                valid_samples.append((qid, exec_time))
    
    if not valid_samples:
        print("No valid training samples found in results")
        return
    
    
    predictions = []
    for qid, actual_time in valid_samples:
        pred_time, _ = knn_predictor.predict(qid)
        predictions.append((qid, actual_time, pred_time))
    
    
    actual_times = [p[1] for p in predictions]
    pred_times = [p[2] for p in predictions]
    
    errors = np.array(actual_times) - np.array(pred_times)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / actual_times)) * 100 if np.all(np.array(actual_times) > 0) else float('inf')
    
    
    correlation = np.corrcoef(actual_times, pred_times)[0, 1]
    
    print(f"  KNN Prediction Performance:")
    print(f"   k: {knn_predictor.k}")
    print(f"   Weights: {knn_predictor.weights}")
    print(f"   Metric: {knn_predictor.metric}")
    print(f"   PCA: {'Enabled' if knn_predictor.use_pca else 'Disabled'}")
    print(f"   Samples: {len(predictions)}")
    print(f"   MAE: {mae:.2f}s")
    print(f"   RMSE: {rmse:.2f}s")
    print(f"   MAPE: {mape:.1f}%")
    print(f"   Correlation: {correlation:.3f}")
    
    
    print(f"\n Sample predictions (first 5):")
    for i in range(min(5, len(predictions))):
        qid, actual, pred = predictions[i]
        print(f"   {i+1}. {qid}: Actual={actual:.1f}s, Pred={pred:.1f}s, Error={actual-pred:.1f}s")
    
    print("="*60)

def _print_final_metrics(final_metrics: Dict[str, Any], batch_stats: Dict[str, Any], args):
    
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    
    print(f" Accuracy: {final_metrics['accuracy']:.4f}")
    print(f" Processed: {final_metrics['successful_count']}/{final_metrics['total_count']} successful")
    
    
    if args.knn_enabled:
        print(f" KNN Prediction: Enabled (k={args.knn_k}, warmup: {args.knn_warmup_ratio*100:.0f}%)")
    
    print(f"\n  TIME STATISTICS:")
    print(f"   Batch total time: {batch_stats['batch_total_time']:.2f}s")
    print(f"   Actual parallel duration: {batch_stats['actual_batch_duration']:.2f}s")
    print(f"   Avg question time: {final_metrics['latency_avg']:.2f}s ± {final_metrics['latency_std']:.2f}s")
    print(f"   Total sequential time: {batch_stats['total_sequential_time']:.2f}s")
    print(f"   Parallel efficiency: {batch_stats['parallel_efficiency']:.2%}")
    
    if batch_stats['actual_batch_duration'] > 0:
        speedup = batch_stats['total_sequential_time'] / batch_stats['actual_batch_duration']
        print(f"   Speedup factor: {speedup:.2f}x")
    
    if final_metrics.get("task_stats"):
        task_stats = final_metrics["task_stats"]
        print(f"\n TASK OPTIMIZATION:")
        print(f"   Original tasks: {task_stats['original_tasks']}")
        print(f"   Executed tasks: {task_stats['executed_tasks']}")
        print(f"   Optimization rate: {task_stats['optimization_rate']:.2%}")

    if final_metrics.get("token_stats"):
        token_stats = final_metrics["token_stats"]
        print(f"\n TOKEN STATISTICS:")
        print(f"   Counted questions: {token_stats['count']}")
        print(f"   Avg LLM calls: {token_stats['avg_calls']:.2f}")
        print(f"   Avg input tokens: {token_stats['avg_input_tokens']:.2f}")
        print(f"   Avg output tokens: {token_stats['avg_output_tokens']:.2f}")
        print(f"   Avg total tokens: {token_stats['avg_total_tokens']:.2f}")
    
    print("="*60)

def get_model_new(model_type, model_name, **kwargs):
    """新的获取模型函数，返回适配器"""
    return create_llm_adapter(model_type, model_name, **kwargs)

async def main():
    
    print("=" * 60)
    print(" DynaCall with Pre-fetched Embeddings")
    print("=" * 60)
    store_dir = os.path.dirname(args.store)
    if store_dir and not os.path.exists(store_dir):
        os.makedirs(store_dir, exist_ok=True)

    
    configs = get_configs(args)
    model_name = args.model_name or configs["default_model"]
    dataset = get_dataset(args)
    tools = get_tools(model_name, args)
    
    
    if args.model_type in ["openai", "azure"]:
        prompt_type = "gpt"
    else:
        prompt_type = "llama"
    
    print(f" Initializing DynaCall...")
    print(f"   Benchmark: {args.benchmark_name}")
    print(f"   Model: {model_name} ({args.model_type})")
    print(f"   Tools: {len(tools)} available")
    effective_max_replans = (
        args.max_replans if args.max_replans is not None else configs["max_replans"]
    )
    print(f"   Max replans: {effective_max_replans}")
    print(f"   KNN Prediction: {'Enabled' if args.knn_enabled else 'Disabled'}")
    if args.knn_enabled:
        print(f"   Embedding model: {args.embedding_model}")
        print(f"   Embedding workers: {args.embedding_workers}")
        print(f"   KNN k: {args.knn_k}, weights: {args.knn_weights}")
        print(f"   KNN warmup ratio: {args.knn_warmup_ratio}")
        print(f"   PCA: {'Enabled' if args.use_pca else 'Disabled'}")
    
    llm = get_model_new(
        model_type=args.model_type,
        model_name=model_name,
        vllm_port=args.vllm_port,
        stream=False,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE")
    )
    planner_llm = get_model_new(
        model_type=args.model_type,
        model_name=model_name,
        vllm_port=args.vllm_port,
        stream=args.use_early_execution,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE")
    )
    prompts = configs["prompts"][prompt_type]

    agent = Controller(
        tools=tools,
        planner_llm=planner_llm,
        planner_example_prompt=prompts["planner_prompt"],
        planner_example_prompt_replan=prompts.get("planner_prompt_replan"),
        planner_stop=None,
        planner_stream=args.use_early_execution,
        agent_llm=llm,
        joinner_prompt=prompts["output_prompt"],
        joinner_prompt_final=prompts.get("output_prompt_final"),
        planner_critic_prompt=prompts.get("planner_critic_prompt"),
        planner_critic_prompt_replan=prompts.get("planner_critic_prompt_replan"),
        max_replans=effective_max_replans,
        benchmark=args.do_benchmark,
    )
    
    
    results = await process_with_prefetched_embeddings(agent, dataset, args)
    
    
    if args.do_benchmark and results:
        print("\n Final benchmark evaluation...")
        
        all_answers = []
        all_labels = []
        all_times = []
        
        for question_id, result in results.items():
            if result.get("status") == "success":
                all_answers.append(result["answer"])
                all_labels.append(result["label"])
                all_times.append(result["time"])
        
        if all_answers:
            scores = []
            for answer, label in zip(all_answers, all_labels):
                if isinstance(label, list):
                    score = max(
                        compare_answer(answer, item, benchmark_name=args.benchmark_name)
                        for item in label
                    ) if label else False
                else:
                    score = compare_answer(answer, label, benchmark_name=args.benchmark_name)
                scores.append(score)
            
            accuracy = np.mean(scores)
            latency_avg = np.mean(all_times)
            latency_std = np.std(all_times)
            
            print(f"Individual question latency: {latency_avg:.2f}s ± {latency_std:.2f}s")
            print(f"Accuracy: {accuracy:.4f}")
            
            
            eval_result = {
                "accuracy": accuracy,
                "latency_avg": latency_avg,
                "latency_std": latency_std,
                "correct_count": sum(scores),
                "total_count": len(scores),
                "knn_enabled": args.knn_enabled,
                "knn_k": args.knn_k if args.knn_enabled else None,
                "knn_warmup_ratio": args.knn_warmup_ratio if args.knn_enabled else None,
                "embedding_model": args.embedding_model if args.knn_enabled else None,
                "use_pca": args.use_pca if args.knn_enabled else None,
                "results": results
            }
            eval_file = args.store.replace(".json", "_eval.json")
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(eval_result, f, ensure_ascii=False, indent=2)
            print(f" Evaluation saved to: {eval_file}")
    
    print(" Processing completed!")

if __name__ == "__main__":
    asyncio.run(main())
