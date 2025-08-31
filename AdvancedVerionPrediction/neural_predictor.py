import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, 
    Concatenate, Embedding, Flatten, LSTM, GRU
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedNeuralPredictor:
    """Advanced neural network predictor for sports betting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.embedding_dims = {}
        
    def create_deep_neural_network(self, input_dim, num_classes, model_type='standard'):
        """Create sophisticated neural network architectures"""
        
        if model_type == 'standard':
            model = Sequential([
                Dense(512, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(64, activation='relu'),
                Dropout(0.2),
                
                Dense(32, activation='relu'),
                
                Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
            ])
            
        elif model_type == 'residual':
            # Residual connections for deeper networks
            inputs = Input(shape=(input_dim,))
            
            # First block
            x = Dense(512, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Residual block 1
            residual = x
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # Add residual connection
            if x.shape[-1] == residual.shape[-1]:
                x = tf.keras.layers.Add()([x, residual])
            
            # Residual block 2
            residual = x
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            
            if x.shape[-1] == residual.shape[-1]:
                x = tf.keras.layers.Add()([x, residual])
            
            # Final layers
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.1)(x)
            outputs = Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
        elif model_type == 'attention':
            # Attention mechanism for feature importance
            inputs = Input(shape=(input_dim,))
            
            # Feature attention
            attention_weights = Dense(input_dim, activation='softmax', name='attention')(inputs)
            attended_features = tf.keras.layers.Multiply()([inputs, attention_weights])
            
            # Main network
            x = Dense(256, activation='relu')(attended_features)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.1)(x)
            
            outputs = Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def create_ensemble_neural_network(self, input_dim, num_classes):
        """Create ensemble of different neural architectures"""
        # Create multiple models
        models = []
        
        # Standard deep network
        model1 = self.create_deep_neural_network(input_dim, num_classes, 'standard')
        models.append(model1)
        
        # Residual network
        model2 = self.create_deep_neural_network(input_dim, num_classes, 'residual')
        models.append(model2)
        
        # Attention network
        model3 = self.create_deep_neural_network(input_dim, num_classes, 'attention')
        models.append(model3)
        
        # Ensemble averaging
        inputs = Input(shape=(input_dim,))
        
        # Get predictions from all models
        predictions = []
        for i, model in enumerate(models):
            pred = model(inputs)
            predictions.append(pred)
        
        # Average predictions
        ensemble_output = tf.keras.layers.Average()(predictions)
        
        ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
        
        return ensemble_model, models
    
    def train_neural_model(self, X_train, y_train, X_val, y_val, target_name):
        """Train neural network with advanced techniques"""
        input_dim = X_train.shape[1]
        
        # Determine number of classes
        if target_name == 'winner':
            num_classes = 3
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            num_classes = 2
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        
        # Create model
        if X_train.shape[0] > 1000:  # Use ensemble for larger datasets
            model, base_models = self.create_ensemble_neural_network(input_dim, num_classes)
        else:
            model = self.create_deep_neural_network(input_dim, num_classes, 'standard')
        
        # Compile model
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint(f'best_model_{target_name}.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        return model, history
    
    def create_lstm_predictor(self, sequence_data, target_name):
        """Create LSTM model for sequential pattern recognition"""
        # Reshape data for LSTM (samples, timesteps, features)
        if len(sequence_data.shape) == 2:
            # Add time dimension
            sequence_data = sequence_data.reshape(sequence_data.shape[0], 1, sequence_data.shape[1])
        
        timesteps = sequence_data.shape[1]
        features = sequence_data.shape[2]
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
            Dropout(0.3),
            
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(3 if target_name == 'winner' else 2, activation='softmax' if target_name == 'winner' else 'sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if target_name == 'winner' else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_advanced_validation(self, X, y, target_name):
        """Train with time-series aware validation"""
        # Time series split for temporal validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        fold_scores = []
        best_model = None
        best_score = -1
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"  Training fold {fold + 1}/5...")
            
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Train neural model
            model, history = self.train_neural_model(
                X_train_fold.values, y_train_fold,
                X_val_fold.values, y_val_fold,
                target_name
            )
            
            # Evaluate
            val_score = model.evaluate(X_val_fold.values, y_val_fold, verbose=0)[1]  # Accuracy
            fold_scores.append(val_score)
            
            if val_score > best_score:
                best_score = val_score
                best_model = model
        
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"  Neural Network CV: {avg_score:.3f} ± {std_score:.3f}")
        
        return best_model

class ReinforcementLearningPredictor:
    """Reinforcement learning component for continuous improvement"""
    
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.action_history = []
        self.reward_history = []
        
    def get_state_representation(self, match_features):
        """Convert match features to state representation"""
        # Discretize continuous features for Q-learning
        state_features = []
        
        # Odds-based state
        if 'draw_winner_diff' in match_features:
            diff = match_features['draw_winner_diff']
            if diff > 2.0:
                state_features.append('high_draw_diff')
            elif diff > 1.0:
                state_features.append('medium_draw_diff')
            else:
                state_features.append('low_draw_diff')
        
        # Market efficiency state
        if 'market_efficiency' in match_features:
            eff = match_features['market_efficiency']
            if eff > 0.7:
                state_features.append('high_efficiency')
            elif eff > 0.4:
                state_features.append('medium_efficiency')
            else:
                state_features.append('low_efficiency')
        
        # Performance difference state
        if 'performance_rating_diff' in match_features:
            diff = match_features['performance_rating_diff']
            if diff > 0.2:
                state_features.append('home_advantage')
            elif diff < -0.2:
                state_features.append('away_advantage')
            else:
                state_features.append('balanced')
        
        return tuple(state_features)
    
    def select_action(self, state, available_actions):
        """Select action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(available_actions)
        else:
            # Exploitation
            q_values = [self.q_table[state][action] for action in available_actions]
            best_action_idx = np.argmax(q_values)
            return available_actions[best_action_idx]
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning algorithm"""
        current_q = self.q_table[state][action]
        
        # Find maximum Q-value for next state
        next_q_values = list(self.q_table[next_state].values())
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, prediction, actual_result, confidence):
        """Calculate reward based on prediction accuracy and confidence"""
        if prediction == actual_result:
            # Correct prediction - reward based on confidence
            base_reward = 1.0
            confidence_bonus = confidence * 0.5  # Up to 0.5 bonus for high confidence
            return base_reward + confidence_bonus
        else:
            # Incorrect prediction - penalty based on confidence
            base_penalty = -1.0
            confidence_penalty = confidence * 0.5  # Higher penalty for confident wrong predictions
            return base_penalty - confidence_penalty
    
    def learn_from_results(self, predictions_with_results):
        """Learn from prediction results using reinforcement learning"""
        for result in predictions_with_results:
            match_features = result['match_features']
            predictions = result['predictions']
            actual_results = result['actual_results']
            confidence_scores = result['confidence_scores']
            
            state = self.get_state_representation(match_features)
            
            # Learn from each prediction
            for target_name in predictions:
                if target_name in actual_results:
                    prediction = predictions[target_name]
                    actual = actual_results[target_name]
                    confidence = confidence_scores.get(target_name, 0.5)
                    
                    # Calculate reward
                    reward = self.calculate_reward(prediction, actual, confidence)
                    
                    # Update Q-value
                    action = f"{target_name}_{prediction}"
                    next_state = state  # Simplified - could be more sophisticated
                    
                    self.update_q_value(state, action, reward, next_state)
                    
                    # Store history
                    self.action_history.append((state, action, prediction, actual))
                    self.reward_history.append(reward)
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def get_rl_confidence_adjustment(self, match_features, target_name, prediction):
        """Get confidence adjustment based on RL learning"""
        state = self.get_state_representation(match_features)
        action = f"{target_name}_{prediction}"
        
        # Get Q-value for this state-action pair
        q_value = self.q_table[state][action]
        
        # Convert Q-value to confidence adjustment (-0.2 to +0.2)
        confidence_adjustment = np.tanh(q_value / 2) * 0.2
        
        return confidence_adjustment

class MetaLearningPredictor:
    """Meta-learning component for learning to learn"""
    
    def __init__(self):
        self.meta_features = []
        self.meta_targets = []
        self.meta_model = None
        self.task_embeddings = {}
        
    def extract_meta_features(self, match_data, model_predictions, model_confidences):
        """Extract meta-features for meta-learning"""
        meta_features = []
        
        # Model agreement features
        if len(model_predictions) > 1:
            predictions_array = np.array(list(model_predictions.values()))
            confidences_array = np.array(list(model_confidences.values()))
            
            # Agreement metrics
            meta_features.extend([
                np.std(predictions_array),  # Prediction variance
                np.mean(confidences_array),  # Average confidence
                np.std(confidences_array),   # Confidence variance
                len(np.unique(predictions_array)) / len(predictions_array)  # Prediction diversity
            ])
        
        # Match complexity features
        if 'market_efficiency' in match_data:
            meta_features.append(match_data['market_efficiency'])
        
        if 'odds_entropy' in match_data:
            meta_features.append(match_data['odds_entropy'])
        
        if 'performance_rating_diff' in match_data:
            meta_features.append(abs(match_data['performance_rating_diff']))
        
        # Pattern complexity
        if 'high_draw_diff_pattern' in match_data:
            meta_features.append(match_data['high_draw_diff_pattern'])
        
        return np.array(meta_features)
    
    def train_meta_model(self, meta_features_list, prediction_accuracies):
        """Train meta-model to predict prediction accuracy"""
        if len(meta_features_list) < 10:
            return None
        
        X_meta = np.array(meta_features_list)
        y_meta = np.array(prediction_accuracies)
        
        # Simple neural network for meta-learning
        meta_model = Sequential([
            Dense(32, activation='relu', input_shape=(X_meta.shape[1],)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')  # Predict accuracy (0-1)
        ])
        
        meta_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train
        meta_model.fit(X_meta, y_meta, epochs=100, validation_split=0.2, verbose=0)
        
        self.meta_model = meta_model
        return meta_model
    
    def predict_prediction_quality(self, match_data, model_predictions, model_confidences):
        """Predict how accurate the predictions will be"""
        if self.meta_model is None:
            return 0.5
        
        meta_features = self.extract_meta_features(match_data, model_predictions, model_confidences)
        
        if len(meta_features) == 0:
            return 0.5
        
        # Reshape for prediction
        meta_features = meta_features.reshape(1, -1)
        
        try:
            predicted_accuracy = self.meta_model.predict(meta_features, verbose=0)[0][0]
            return float(predicted_accuracy)
        except:
            return 0.5

class UltraAdvancedSportsPredictor:
    """Ultra-advanced sports predictor combining all techniques"""
    
    def __init__(self, train_folder='train', results_folder='previousMatch'):
        # Initialize base predictor
        self.base_predictor = AdvancedSportsPredictor(train_folder, results_folder)
        
        # Advanced components
        self.neural_predictor = AdvancedNeuralPredictor()
        self.rl_predictor = ReinforcementLearningPredictor()
        self.meta_learner = MetaLearningPredictor()
        
        # Ensemble weights
        self.ensemble_weights = {
            'traditional_ml': 0.4,
            'neural_network': 0.3,
            'reinforcement_learning': 0.2,
            'meta_learning': 0.1
        }
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_history = defaultdict(list)
        
    def train_ultra_advanced_models(self):
        """Train all model types"""
        print("=== TRAINING ULTRA-ADVANCED PREDICTION SYSTEM ===")
        
        # 1. Train traditional ML models
        print("1. Training traditional ML ensemble...")
        self.base_predictor.train_advanced_models()
        
        # 2. Train neural networks
        if TF_AVAILABLE and not self.base_predictor.matched_data.empty:
            print("2. Training neural networks...")
            self.train_neural_networks()
        
        # 3. Initialize reinforcement learning
        print("3. Initializing reinforcement learning...")
        # RL will learn from prediction results over time
        
        # 4. Train meta-learner if enough data
        print("4. Training meta-learner...")
        if len(self.prediction_history) > 50:
            self.train_meta_learner()
        
        print("Ultra-advanced training completed!")
    
    def train_neural_networks(self):
        """Train neural network components"""
        # Prepare data
        X, _ = self.base_predictor.prepare_advanced_features(self.base_predictor.matched_data, is_training=True)
        targets = self.base_predictor.create_advanced_targets(self.base_predictor.matched_data)
        
        for target_name, y in targets.items():
            if target_name.endswith('_weights'):
                continue
            
            if len(np.unique(y)) < 2:
                continue
            
            print(f"  Training neural network for {target_name}...")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train neural model
            neural_model, history = self.neural_predictor.train_neural_model(
                X_train.values, y_train, X_val.values, y_val, target_name
            )
            
            if neural_model:
                self.neural_predictor.models[target_name] = neural_model
                
                # Evaluate
                val_score = neural_model.evaluate(X_val.values, y_val, verbose=0)[1]
                print(f"    Neural network validation accuracy: {val_score:.3f}")
    
    def train_meta_learner(self):
        """Train meta-learner from prediction history"""
        if len(self.prediction_history) < 10:
            return
        
        meta_features_list = []
        accuracy_list = []
        
        for pred_record in self.prediction_history:
            if 'meta_features' in pred_record and 'accuracy' in pred_record:
                meta_features_list.append(pred_record['meta_features'])
                accuracy_list.append(pred_record['accuracy'])
        
        if len(meta_features_list) >= 10:
            self.meta_learner.train_meta_model(meta_features_list, accuracy_list)
            print("Meta-learner trained successfully")
    
    def ultra_predict(self, home_team, away_team, league, upcoming_filename=None):
        """Ultra-advanced prediction combining all techniques"""
        print(f"\n{'='*70}")
        print(f"ULTRA-ADVANCED PREDICTION SYSTEM")
        print(f"{'='*70}")
        
        # 1. Get base prediction
        base_result = self.base_predictor.predict_with_advanced_analysis(
            home_team, away_team, league, upcoming_filename
        )
        
        if not base_result:
            return None
        
        # 2. Get neural network predictions
        neural_predictions = {}
        neural_confidences = {}
        
        if TF_AVAILABLE and self.neural_predictor.models:
            neural_predictions, neural_confidences = self.get_neural_predictions(
                base_result['match_data']
            )
        
        # 3. Get RL adjustments
        rl_adjustments = {}
        for target_name, prediction in base_result['predictions'].items():
            rl_adj = self.rl_predictor.get_rl_confidence_adjustment(
                base_result['match_data'], target_name, prediction
            )
            rl_adjustments[target_name] = rl_adj
        
        # 4. Get meta-learning quality prediction
        meta_quality = self.meta_learner.predict_prediction_quality(
            base_result['match_data'],
            base_result['predictions'],
            base_result['confidence']
        )
        
        # 5. Combine all predictions
        final_predictions = self.combine_all_predictions(
            base_result, neural_predictions, rl_adjustments, meta_quality
        )
        
        # 6. Display ultra-advanced results
        self.display_ultra_results(final_predictions, base_result, neural_predictions, meta_quality)
        
        return final_predictions
    
    def get_neural_predictions(self, match_data):
        """Get predictions from neural networks"""
        predictions = {}
        confidences = {}
        
        # Prepare features for neural network
        match_df = pd.DataFrame([match_data])
        X, _ = self.base_predictor.prepare_advanced_features(match_df, is_training=False)
        
        if X.empty:
            return predictions, confidences
        
        for target_name, model in self.neural_predictor.models.items():
            try:
                # Get prediction
                pred_proba = model.predict(X.values, verbose=0)[0]
                
                if len(pred_proba) > 2:  # Multi-class
                    prediction = np.argmax(pred_proba)
                    confidence = np.max(pred_proba)
                else:  # Binary
                    prediction = 1 if pred_proba[0] > 0.5 else 0
                    confidence = max(pred_proba[0], 1 - pred_proba[0])
                
                predictions[target_name] = prediction
                confidences[target_name] = confidence
                
            except Exception as e:
                print(f"Error in neural prediction for {target_name}: {e}")
        
        return predictions, confidences
    
    def combine_all_predictions(self, base_result, neural_predictions, rl_adjustments, meta_quality):
        """Combine predictions from all models using advanced weighting"""
        combined_predictions = {}
        combined_confidences = {}
        
        for target_name in base_result['predictions']:
            # Get predictions from different sources
            base_pred = base_result['predictions'][target_name]
            base_conf = base_result['confidence'][target_name]
            
            neural_pred = neural_predictions.get(target_name, base_pred)
            neural_conf = neural_predictions.get(target_name, base_conf)
            
            rl_adjustment = rl_adjustments.get(target_name, 0)
            
            # Weighted combination
            if base_pred == neural_pred:
                # Models agree - boost confidence
                final_prediction = base_pred
                final_confidence = (base_conf * self.ensemble_weights['traditional_ml'] + 
                                  neural_conf * self.ensemble_weights['neural_network']) / \
                                 (self.ensemble_weights['traditional_ml'] + self.ensemble_weights['neural_network'])
                final_confidence = min(0.95, final_confidence * 1.1)  # Agreement bonus
            else:
                # Models disagree - use base prediction but reduce confidence
                final_prediction = base_pred
                final_confidence = base_conf * 0.8  # Disagreement penalty
            
            # Apply RL adjustment
            final_confidence = max(0.05, min(0.95, final_confidence + rl_adjustment))
            
            # Apply meta-learning quality adjustment
            final_confidence = final_confidence * meta_quality
            
            combined_predictions[target_name] = final_prediction
            combined_confidences[target_name] = final_confidence
        
        return {
            'predictions': combined_predictions,
            'confidence': combined_confidences,
            'base_result': base_result,
            'neural_predictions': neural_predictions,
            'rl_adjustments': rl_adjustments,
            'meta_quality': meta_quality
        }
    
    def display_ultra_results(self, final_result, base_result, neural_predictions, meta_quality):
        """Display ultra-advanced prediction results"""
        print(f"\n🚀 ULTRA-ADVANCED PREDICTION RESULTS:")
        print(f"   Meta-Learning Quality Score: {meta_quality:.1%}")
        
        # Winner prediction
        if 'winner' in final_result['predictions']:
            winner_pred = final_result['predictions']['winner']
            confidence = final_result['confidence']['winner']
            
            home_team = base_result['match_data']['home_team']
            away_team = base_result['match_data']['away_team']
            winner_map = {0: away_team, 1: 'Draw', 2: home_team}
            predicted_winner = winner_map.get(winner_pred, 'Unknown')
            
            print(f"\n🏆 FINAL WINNER PREDICTION:")
            print(f"   Winner: {predicted_winner}")
            print(f"   Ultra-Confidence: {confidence:.1%}")
            
            # Show model agreement
            base_pred = base_result['predictions'].get('winner', -1)
            neural_pred = neural_predictions.get('winner', -1)
            
            if base_pred == neural_pred == winner_pred:
                print(f"   ✅ ALL MODELS AGREE - High reliability")
            elif base_pred == winner_pred:
                print(f"   ⚠️  Traditional ML confident, Neural disagrees")
            else:
                print(f"   ⚠️  Model disagreement - proceed with caution")
        
        # Advanced recommendations
        print(f"\n🎯 ULTRA-ADVANCED RECOMMENDATIONS:")
        
        recommendations = []
        
        # Confidence-based recommendations
        avg_confidence = np.mean(list(final_result['confidence'].values()))
        
        if avg_confidence > 0.85:
            recommendations.append("ULTRA-STRONG BET: Exceptional confidence across all models")
        elif avg_confidence > 0.75:
            recommendations.append("STRONG BET: High confidence with model agreement")
        elif avg_confidence > 0.65:
            recommendations.append("GOOD BET: Solid confidence levels")
        elif avg_confidence > 0.55:
            recommendations.append("MODERATE BET: Reasonable confidence")
        else:
            recommendations.append("AVOID: Low confidence predictions")
        
        # Meta-learning recommendations
        if meta_quality > 0.8:
            recommendations.append("HIGH QUALITY: Meta-learner indicates reliable predictions")
        elif meta_quality < 0.4:
            recommendations.append("CAUTION: Meta-learner indicates uncertain predictions")
        
        # Pattern-based recommendations
        if base_result.get('pattern_insights', {}).get('high_draw_diff_pattern', {}).get('triggered', False):
            recommendations.append("PATTERN DETECTED: Your specific draw-difference rule activated")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def save_ultra_models(self, filename='ultra_advanced_predictor.pkl'):
        """Save all ultra-advanced models"""
        # Save base predictor
        self.base_predictor.save_advanced_models('base_' + filename)
        
        # Save additional components
        ultra_data = {
            'neural_predictor': self.neural_predictor,
            'rl_predictor': self.rl_predictor,
            'meta_learner': self.meta_learner,
            'ensemble_weights': self.ensemble_weights,
            'prediction_history': self.prediction_history,
            'accuracy_history': dict(self.accuracy_history)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(ultra_data, f)
        
        print(f"Ultra-advanced models saved to {filename}")
    
    def load_ultra_models(self, filename='ultra_advanced_predictor.pkl'):
        """Load all ultra-advanced models"""
        # Load base predictor
        base_loaded = self.base_predictor.load_advanced_models('base_' + filename)
        
        try:
            with open(filename, 'rb') as f:
                ultra_data = pickle.load(f)
            
            self.neural_predictor = ultra_data.get('neural_predictor', AdvancedNeuralPredictor())
            self.rl_predictor = ultra_data.get('rl_predictor', ReinforcementLearningPredictor())
            self.meta_learner = ultra_data.get('meta_learner', MetaLearningPredictor())
            self.ensemble_weights = ultra_data.get('ensemble_weights', self.ensemble_weights)
            self.prediction_history = ultra_data.get('prediction_history', [])
            self.accuracy_history = defaultdict(list, ultra_data.get('accuracy_history', {}))
            
            print("Ultra-advanced models loaded successfully")
            return True
            
        except FileNotFoundError:
            print("Ultra-advanced model file not found")
            return base_loaded
        except Exception as e:
            print(f"Error loading ultra-advanced models: {e}")
            return base_loaded

def main():
    """Main execution for ultra-advanced predictor"""
    predictor = UltraAdvancedSportsPredictor()
    
    # Load or train models
    if not predictor.load_ultra_models():
        print("Training new ultra-advanced models...")
        predictor.train_ultra_advanced_models()
        predictor.save_ultra_models()
    
    print(f"\n{'='*70}")
    print(f"🚀 ULTRA-ADVANCED SPORTS PREDICTION SYSTEM")
    print(f"{'='*70}")
    print("Features:")
    print("✅ Advanced ensemble learning with 10+ algorithms")
    print("✅ Deep neural networks with attention mechanisms")
    print("✅ Reinforcement learning for continuous improvement")
    print("✅ Meta-learning for prediction quality assessment")
    print("✅ Your specific draw-difference pattern detection")
    print("✅ Advanced pattern recognition and market analysis")
    print("✅ Hyperparameter optimization with Optuna")
    print("✅ Sophisticated feature engineering")
    print("✅ Multi-model ensemble with intelligent weighting")
    
    print("\nCommands:")
    print("1. ultra_predict <home> <away> <league> [upcoming_file] - Ultra prediction")
    print("2. train - Train all models")
    print("3. save - Save models")
    print("4. performance - Show performance metrics")
    print("5. quit")
    
    while True:
        try:
            command = input("\nEnter command: ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            
            if parts[0] == 'quit':
                break
            elif parts[0] == 'ultra_predict' and len(parts) >= 4:
                home_team = parts[1]
                away_team = parts[2]
                league = parts[3]
                upcoming_file = parts[4] if len(parts) > 4 else None
                
                predictor.ultra_predict(home_team, away_team, league, upcoming_file)
                
            elif parts[0] == 'train':
                predictor.train_ultra_advanced_models()
                predictor.save_ultra_models()
                
            elif parts[0] == 'save':
                predictor.save_ultra_models()
                
            elif parts[0] == 'performance':
                print(predictor.base_predictor.get_model_performance_report())
                
            else:
                print("Invalid command")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()