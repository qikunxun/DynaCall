import os
import time
import os
import asyncio
import threading
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import openai
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingManager:
    
    
    def __init__(self, api_key: str = None, cache_path: str = None, model: str = "text-embedding-3-small",
                 embedding_batch_size: int = 10, embedding_workers: int = 10):
        self.client = openai.OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        self.model = model
        self.embeddings_cache = {}  # question_id -> embedding
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.embedding_batch_size = embedding_batch_size
        self.embedding_workers = embedding_workers 
        print(f" Embedding Manager: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        
        try:
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            return [np.array(item.embedding, dtype=np.float32) for item in response.data]
        except Exception as e:
            print(f"  Batch embedding error: {e}")
            
            return [np.zeros(1536, dtype=np.float32) for _ in texts]
    
    async def prefetch_embeddings(self, questions: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        
        print(f" Pre-fetching embeddings for {len(questions)} questions...")
        if os.path.exists(self.cache_path):
            embeddings_cache = np.load(self.cache_path, allow_pickle=True).item()
            with self.lock:
                self.embeddings_cache.update(embeddings_cache)
            print(f"   Loaded {len(embeddings_cache)} cached embeddings from {self.cache_path}")
            return embeddings_cache
        
        start_time = time.time()
        
        
        texts = [q['question'] for q in questions]
        ids = [q['id'] for q in questions]
        
        
        tasks = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch_texts = texts[i:i + self.embedding_batch_size]
            batch_ids = ids[i:i + self.embedding_batch_size]
            tasks.append((batch_ids, batch_texts))
        
        
        semaphore = asyncio.Semaphore(self.embedding_workers)
        
        async def process_batch(batch_ids, batch_texts):
            async with semaphore:
                batch_embeddings = await self._get_batch_embeddings(batch_texts)
                return zip(batch_ids, batch_embeddings)
        
        
        batch_results = await asyncio.gather(
            *[process_batch(batch_ids, batch_texts) for batch_ids, batch_texts in tasks]
        )
        
        
        embeddings_cache = {}
        total_embeddings = 0
        for batch_result in batch_results:
            for qid, embedding in batch_result:
                embeddings_cache[qid] = embedding
                total_embeddings += 1
        
        elapsed = time.time() - start_time
        print(f" Pre-fetched {total_embeddings} embeddings in {elapsed:.2f}s "
              f"({total_embeddings/elapsed:.1f} embeddings/sec)")
        
        with self.lock:
            self.embeddings_cache.update(embeddings_cache)
        np.save(self.cache_path, embeddings_cache)
        print(embeddings_cache)
        return embeddings_cache
    
    def get_embedding(self, question_id: str) -> Optional[np.ndarray]:
        
        with self.lock:
            return self.embeddings_cache.get(question_id)
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        
        with self.lock:
            return self.embeddings_cache.copy()


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