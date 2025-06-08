# Carbon Emissions Forecasting Model for SDG 13: Climate Action
# Author: [Your Name]
# Purpose: Predict CO2 emissions using socio-economic indicators

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CarbonEmissionsPredictor:
    """
    A comprehensive model to predict CO2 emissions for SDG 13: Climate Action
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.results = {}
        
    def load_and_prepare_data(self, file_path=None):
        """
        Load and prepare the emissions dataset
        For demo purposes, we'll create synthetic data based on real patterns
        """
        if file_path:
            # Load real data if file path provided
            self.data = pd.read_csv(file_path)
        else:
            # Create synthetic dataset for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic features based on real-world relationships
            gdp_per_capita = np.random.lognormal(8, 1.5, n_samples)
            population = np.random.lognormal(15, 2, n_samples)
            energy_consumption = np.random.lognormal(6, 1, n_samples)
            industrial_production = np.random.normal(50, 20, n_samples)
            renewable_energy_pct = np.random.beta(2, 5, n_samples) * 100
            
            # Generate target variable with realistic relationships
            co2_emissions = (
                0.5 * np.log(gdp_per_capita) +
                0.3 * np.log(population) +
                0.4 * np.log(energy_consumption) +
                0.002 * industrial_production -
                0.01 * renewable_energy_pct +
                np.random.normal(0, 0.5, n_samples)
            )
            
            self.data = pd.DataFrame({
                'gdp_per_capita': gdp_per_capita,
                'population': population,
                'energy_consumption': energy_consumption,
                'industrial_production': industrial_production,
                'renewable_energy_pct': renewable_energy_pct,
                'co2_emissions': co2_emissions,
                'year': np.random.randint(2000, 2021, n_samples),
                'country': np.random.choice(['USA', 'China', 'India', 'Germany', 'Brazil'], n_samples)
            })
        
        print(f"Dataset loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        print("\nDataset Info:")
        print(self.data.info())
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print("\nDescriptive Statistics:")
        print(self.data.describe())
        
        # Correlation analysis
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
        axes[0,0].set_title('Feature Correlation Matrix')
        
        # Distribution of target variable
        axes[0,1].hist(self.data['co2_emissions'], bins=30, alpha=0.7, color='skyblue')
        axes[0,1].set_title('Distribution of CO2 Emissions')
        axes[0,1].set_xlabel('CO2 Emissions')
        axes[0,1].set_ylabel('Frequency')
        
        # Scatter plot: GDP vs Emissions
        axes[1,0].scatter(self.data['gdp_per_capita'], self.data['co2_emissions'], alpha=0.6)
        axes[1,0].set_title('GDP per Capita vs CO2 Emissions')
        axes[1,0].set_xlabel('GDP per Capita')
        axes[1,0].set_ylabel('CO2 Emissions')
        
        # Renewable energy impact
        axes[1,1].scatter(self.data['renewable_energy_pct'], self.data['co2_emissions'], alpha=0.6, color='green')
        axes[1,1].set_title('Renewable Energy % vs CO2 Emissions')
        axes[1,1].set_xlabel('Renewable Energy %')
        axes[1,1].set_ylabel('CO2 Emissions')
        
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def preprocess_data(self):
        """
        Clean and preprocess the data
        """
        print("\n=== DATA PREPROCESSING ===")
        
        # Handle missing values
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('co2_emissions')  # Remove target variable
        
        # Impute missing values
        self.data[numeric_features] = self.imputer.fit_transform(self.data[numeric_features])
        
        # Encode categorical variables
        le = LabelEncoder()
        if 'country' in self.data.columns:
            self.data['country_encoded'] = le.fit_transform(self.data['country'])
            numeric_features.append('country_encoded')
        
        # Prepare features and target
        X = self.data[numeric_features]
        y = self.data['co2_emissions']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.feature_names = numeric_features
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features: {self.feature_names}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self):
        """
        Train multiple regression models
        """
        print("\n=== MODEL TRAINING ===")
        
        # Define models
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0)
        }
        
        # Train and evaluate each model
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_test
            }
            
            print(f"R² Score: {test_r2:.4f}")
            print(f"MAE: {test_mae:.4f}")
            print(f"RMSE: {test_rmse:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def compare_models(self):
        """
        Compare model performance
        """
        print("\n=== MODEL COMPARISON ===")
        
        # Create comparison DataFrame
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Test R²': metrics['test_r2'],
                'Test MAE': metrics['test_mae'],
                'Test RMSE': metrics['test_rmse'],
                'CV Score': metrics['cv_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test R²', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score comparison
        axes[0,0].bar(comparison_df['Model'], comparison_df['Test R²'])
        axes[0,0].set_title('R² Score Comparison')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0,1].bar(comparison_df['Model'], comparison_df['Test MAE'])
        axes[0,1].set_title('Mean Absolute Error Comparison')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        best_model_name = comparison_df.iloc[0]['Model']
        best_predictions = self.results[best_model_name]['predictions']
        
        axes[1,0].scatter(self.y_test, best_predictions, alpha=0.6)
        axes[1,0].plot([self.y_test.min(), self.y_test.max()], 
                      [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1,0].set_title(f'{best_model_name}: Predicted vs Actual')
        axes[1,0].set_xlabel('Actual CO2 Emissions')
        axes[1,0].set_ylabel('Predicted CO2 Emissions')
        
        # Residuals plot
        residuals = self.y_test - best_predictions
        axes[1,1].scatter(best_predictions, residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_title(f'{best_model_name}: Residuals Plot')
        axes[1,1].set_xlabel('Predicted Values')
        axes[1,1].set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df, best_model_name
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance for tree-based models
        """
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get feature importance from Random Forest
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importances = rf_model.feature_importances_
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("Feature Importance (Random Forest):")
            print(feature_importance_df.to_string(index=False, float_format='%.4f'))
            
            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.title('Feature Importance for CO2 Emissions Prediction')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
    
    def make_predictions(self, new_data, model_name=None):
        """
        Make predictions on new data
        """
        if model_name is None:
            # Use the best performing model
            comparison_df, best_model_name = self.compare_models()
            model_name = best_model_name
        
        model = self.models[model_name]
        scaled_data = self.scaler.transform(new_data)
        predictions = model.predict(scaled_data)
        
        return predictions
    
    def generate_report(self):
        """
        Generate a comprehensive report
        """
        print("\n" + "="*60)
        print("CARBON EMISSIONS FORECASTING MODEL - FINAL REPORT")
        print("SDG 13: Climate Action")
        print("="*60)
        
        # Dataset overview
        print(f"\nDataset Overview:")
        print(f"- Total samples: {self.data.shape[0]}")
        print(f"- Features: {len(self.feature_names)}")
        print(f"- Target: CO2 Emissions")
        
        # Best model performance
        comparison_df, best_model_name = self.compare_models()
        best_metrics = self.results[best_model_name]
        
        print(f"\nBest Performing Model: {best_model_name}")
        print(f"- R² Score: {best_metrics['test_r2']:.4f}")
        print(f"- Mean Absolute Error: {best_metrics['test_mae']:.4f}")
        print(f"- Root Mean Square Error: {best_metrics['test_rmse']:.4f}")
        print(f"- Cross-validation Score: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
        
        # Key insights
        print(f"\nKey Insights:")
        correlation_matrix = self.explore_data()
        top_correlations = correlation_matrix['co2_emissions'].abs().sort_values(ascending=False)[1:4]
        
        print("Top factors influencing CO2 emissions:")
        for feature, correlation in top_correlations.items():
            print(f"- {feature}: {correlation:.3f} correlation")
        
        print(f"\nModel Implications for SDG 13:")
        print("- This model can help policymakers predict emissions based on economic indicators")
        print("- Identifies key factors that drive carbon emissions")
        print("- Can simulate the impact of policy changes on future emissions")
        print("- Supports evidence-based climate action planning")

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the predictor
    predictor = CarbonEmissionsPredictor()
    
    # Load and explore data
    data = predictor.load_and_prepare_data()
    correlation_matrix = predictor.explore_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Compare models
    comparison_df, best_model = predictor.compare_models()
    
    # Analyze feature importance
    feature_importance_df = predictor.feature_importance_analysis()
    
    # Generate final report
    predictor.generate_report()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE - Ready for SDG 13 Climate Action!")
    print("="*60)
