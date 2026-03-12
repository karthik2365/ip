import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import datetime

# Handle warnings gracefully
warnings.filterwarnings('ignore')

# Seed for reproducibility
np.random.seed(42)

def main():
    print("\n--- Phase 1: Data Generation ---")
    # 1. DATA GENERATION
    days = 365
    bins = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4']
    start_date = datetime.date(2023, 1, 1) # Arbitrary start date

    data_list = []
    for bin_id in bins:
        fill_level = np.random.uniform(5, 10) # start around 5-10%
        for i in range(days):
            current_date = start_date + datetime.timedelta(days=i)
            day_of_week = current_date.weekday() # 0 is Monday, 6 is Sunday
            month = current_date.month
            
            # Base daily increase
            daily_increase = np.random.uniform(2, 5) # 2% to 5% baseline
            
            # Higher fill rates on weekdays
            if day_of_week >= 5: # Weekend
                daily_increase *= 0.5
            else:
                daily_increase *= 1.2
                
            # Seasonal variations (summer more waste)
            if 6 <= month <= 8: # Summer months
                daily_increase *= 1.4
            elif 12 == month or month <= 2: # Winter months
                daily_increase *= 0.8
                
            # Add noise
            noise = np.random.normal(0, 1)
            daily_increase += noise
            
            fill_level += max(0, daily_increase) # ensure no negative increase unless emptied
            
            # Random collection collection events
            if fill_level >= 85:
                # Collection happens, resetting to ~5%
                fill_level = np.random.uniform(3, 7)
                
            # Ensure fill level doesn't exceed 100
            fill_level = min(fill_level, 100)
            
            data_list.append({
                'date': current_date,
                'bin_id': bin_id,
                'fill_level': fill_level,
                'day_of_week': day_of_week,
                'month': month
            })

    df = pd.DataFrame(data_list)
    print("Synthetic data generated successfully.")
    print(f"Total records: {len(df)}")
    print(df.head())

    print("\n--- Phase 2: Data Preprocessing ---")
    # 2. DATA PREPROCESSING
    # Use one bin for training
    df_bin1 = df[df['bin_id'] == 'Bin_1'].copy()
    df_bin1.sort_values('date', inplace=True)
    df_bin1.reset_index(drop=True, inplace=True)

    # Normalize fill levels
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_bin1[['fill_level']])

    # Create sequences (14 days -> 1 day)
    seq_length = 14
    X = []
    y = []

    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:(i + seq_length), 0])
        y.append(scaled_data[i + seq_length, 0])

    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM: (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into 80% training, 20% testing
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

    print("\n--- Phase 3: Model Architecture Configuration ---")
    # 3. MODEL ARCHITECTURE
    model = Sequential()
    model.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    print("Model created successfully. Summary:")
    model.summary()

    print("\n--- Phase 4: Training ---")
    # 4. TRAINING
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.20,
        verbose=1
    )

    print("\n--- Phase 5: Evaluation ---")
    # 5. EVALUATION
    # Predict on train and test
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)

    # Inverse transform predictions and actuals
    y_train_pred = scaler.inverse_transform(y_train_pred_scaled)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)

    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics calculation
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)

    print(f"\nTraining Metrics:")
    print(f"RMSE: {train_rmse:.2f}%")
    print(f"MAE:  {train_mae:.2f}%")
    print(f"R²:   {train_r2:.4f}")

    print(f"\nTesting Metrics:")
    print(f"RMSE: {test_rmse:.2f}%")
    print(f"MAE:  {test_mae:.2f}%")
    print(f"R²:   {test_r2:.4f}")

    print("\nInterpretation:")
    if test_r2 > 0.85:
        acc_desc = "Excellent"
        print("Model accuracy is Excellent. It captures the variance well.")
    elif test_r2 > 0.75:
        acc_desc = "Good"
        print("Model accuracy is Good. It performs reliably.")
    else:
        acc_desc = "Fair"
        print("Model accuracy is Fair. Could be improved with more data or tuning.")

    print("\n--- Phase 6: Visualizations ---")
    # 6. VISUALIZATIONS
    plt.figure(figsize=(16, 10))

    # Chart 1: Training loss vs Validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Chart 2: Train set - Actual vs Predicted
    plt.subplot(2, 2, 2)
    plt.plot(y_train_actual, label='Actual Fill Level', alpha=0.7)
    plt.plot(y_train_pred, label='Predicted Fill Level', alpha=0.8)
    plt.title('Training Set: Actual vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Fill Level (%)')
    plt.legend()

    # Chart 3: Test set - Actual vs Predicted with error shading
    plt.subplot(2, 2, 3)
    plt.plot(y_test_actual, label='Actual Level', alpha=0.7)
    plt.plot(y_test_pred, label='Predicted Level', alpha=0.8)
    errors = np.abs(y_test_actual - y_test_pred)
    plt.fill_between(range(len(y_test_actual)), 
                     (y_test_pred - errors).flatten(), 
                     (y_test_pred + errors).flatten(), 
                     color='gray', alpha=0.2, label='Error Range')
    plt.title('Testing Set: Actual vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Fill Level (%)')
    plt.legend()

    # Chart 4: Histogram of prediction errors
    plt.subplot(2, 2, 4)
    plt.hist((y_test_actual - y_test_pred), bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Prediction Errors (Test Set)')
    plt.xlabel('Prediction Error (%)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('lstm_predictions.png', dpi=300)
    print("Saved 4-panel visualization to 'lstm_predictions.png'")

    print("\n--- Phase 7: Future Forecasting ---")
    # 7. FUTURE FORECASTING
    future_days = 30
    last_14_days = scaled_data[-14:] 
    current_seq = last_14_days.reshape((1, 14, 1))

    future_preds_scaled = []
    for _ in range(future_days):
        pred = model.predict(current_seq, verbose=0)
        future_preds_scaled.append(pred[0, 0])
        
        # Update sequence
        current_seq = np.append(current_seq[0, 1:], pred[0, 0]).reshape(1, 14, 1)

    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    historical_60_days = scaler.inverse_transform(scaled_data[-60:]).flatten()

    plt.figure(figsize=(12, 6))
    # Plot last 60 days
    historical_x = range(1, 61)
    plt.plot(historical_x, historical_60_days, label='Last 60 Days Historical Data', color='blue')

    # Plot next 30 days
    future_x = range(60, 60 + future_days)
    # Start future line from last historical point for continuity
    future_line = np.insert(future_preds, 0, historical_60_days[-1])
    plt.plot(range(60, 60 + future_days + 1), future_line, label='30-Day Forecast', 
             color='orange', linestyle='--', marker='o', markersize=4)

    plt.axhline(y=85, color='red', linestyle='-', label='Collection Threshold (85%)')
    plt.fill_between(range(1, 60 + future_days + 1), 85, 100, color='red', alpha=0.1, label='Collection Risk Zone')

    plt.title('Waste Bin Fill Level: Historical & 30-Day Forecast')
    plt.xlabel('Days')
    plt.ylabel('Fill Level (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('future_forecast.png', dpi=300)
    print("Saved 30-day forecast visualization to 'future_forecast.png'")

    print("\n--- Phase 8: Actionable Insights ---")
    # 8. ACTIONABLE INSIGHTS
    print("Collection Alerts (Next 30 Days):")
    alerts = []
    last_date = df_bin1['date'].iloc[-1]

    for i, pred_level in enumerate(future_preds):
        if pred_level > 80:
            alert_date = last_date + datetime.timedelta(days=i+1)
            alerts.append((alert_date, pred_level))
            print(f" - ALARM: Date {alert_date} | Predicted Fill: {pred_level:.1f}%")

    if not alerts:
        print(" - No critical fill levels (>80%) predicted in the next 30 days.")

    print(f"\nModel Confidence: {acc_desc} (R² = {test_r2:.2f})")

    print("\n5 Recommendations for Waste Management Optimization:")
    print("1. Dynamic Scheduling: Dispatch collection trucks only on dates with an alerted predicted fill level (>80%) rather than a static weekly route.")
    print("2. Resource Allocation: Shift personnel downshift on weekends where prediction models consistently show flat and lower fill rates.")
    print("3. Pre-Weekend Clearance: Ensure bins are cleared before Fridays, as weekend accumulation rates differ and any overflow early in the week stems from weekend backlog.")
    print("4. Seasonal Container Expansion: Prepare extra bin placements or larger capacity during summer months (June-August) to accommodate increased baseline waste.")
    print("5. Predictive Maintenance: Use sensor noise trends to identify possibly faulty sensors that appear to report highly erratic growth outside 2-5% expected daily change.")

    print("\n--- Phase 9: Model Persistence ---")
    # 9. MODEL PERSISTENCE
    model.save('waste_level_lstm_model.h5')
    print("Model saved as 'waste_level_lstm_model.h5'")
    print("To load and use the model for new predictions:")
    print("  from tensorflow.keras.models import load_model")
    print("  loaded_model = load_model('waste_level_lstm_model.h5')")
    print("  predictions = loaded_model.predict(new_14_day_sequence)")

    print("\n--- Phase 10: Final Summary ---")
    # 10. FINAL SUMMARY
    print("="*40)
    print("PROJECT SUMMARY")
    print("="*40)
    print("Model Type:       LSTM Neural Network")
    print(f"Hidden Layers:    {len(model.layers)}")
    print(f"Total Parameters: {model.count_params()}")
    print(f"Test RMSE:        {test_rmse:.2f}%")
    print(f"Test R² Score:    {test_r2:.4f}")
    print("Files Created:")
    print("  - lstm_predictions.png")
    print("  - future_forecast.png")
    print("  - waste_level_lstm_model.h5")
    print("\nNext Steps:")
    print("  - Integrate this script into your scheduling Cron job.")
    print("  - Build a live dashboard to query predictions.")
    print("="*40)
    print("Run Completed End-to-End.")

if __name__ == "__main__":
    main()
