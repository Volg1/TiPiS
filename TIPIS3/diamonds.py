import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

class DiamondPricePredictor:
    def __init__(self):
        self.label_encoders = {}
        self.preprocessor = None
        self.model = None
        self.feature_columns = None
        self.categorical_cols = ['cut', 'color', 'clarity']
        
    def load_and_explore_data(self, filepath):
        """Загрузка и первичный анализ данных"""
        print("="*60)
        print("ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
        print("="*60)
        
        df = pd.read_csv(filepath)
        
        print(f"\nРазмер данных: {df.shape}")
        print(f"\nТипы данных:")
        print(df.dtypes)
        
        print(f"\nПервые 5 строк:")
        print(df.head())
        
        print(f"\nСтатистика числовых признаков:")
        print(df.describe())
        
        return df
    
    def check_missing_values(self, df):
        """Проверка пропущенных значений"""
        print("\n" + "="*60)
        print("ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
        print("="*60)
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Пропущено': missing,
            'Процент': missing_pct
        })
        
        missing_with_data = missing_df[missing_df['Пропущено'] > 0]
        
        if len(missing_with_data) > 0:
            print("\nНайдены пропущенные значения:")
            print(missing_with_data)
        else:
            print("\n✓ Пропущенных значений не найдено")
    
    def analyze_categorical(self, df):
        """Анализ категориальных переменных"""
        print("\n" + "="*60)
        print("АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
        print("="*60)
        
        for col in self.categorical_cols:
            if col in df.columns:
                print(f"\n{col.upper()}:")
                print(f"Уникальные значения: {sorted(df[col].unique())}")
                print(f"Количество уникальных: {df[col].nunique()}")
                print(f"\nРаспределение:")
                print(df[col].value_counts())
                print(f"\nСредняя цена по категориям:")
                print(df.groupby(col)['price'].agg(['mean', 'count', 'std']).round(2))
    
    def encode_categorical(self, df, fit=True):
        """Кодирование категориальных переменных"""
        print("\n" + "="*60)
        print("КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
        print("="*60)
        
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col in df.columns:
                if fit:
                    # Сортируем категории для консистентности
                    unique_values = sorted(df[col].unique())
                    le = LabelEncoder()
                    le.fit(unique_values)
                    self.label_encoders[col] = le
                    print(f"✓ Создан энкодер для {col} с {len(unique_values)} категориями")
                
                # Преобразуем с обработкой неизвестных категорий
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    
                    # Обработка неизвестных категорий
                    def safe_transform(x):
                        if x in le.classes_:
                            return le.transform([x])[0]
                        else:
                            # Используем наиболее частую категорию
                            return le.transform([le.classes_[0]])[0]
                    
                    df_encoded[col + '_encoded'] = df[col].apply(safe_transform)
                else:
                    raise ValueError(f"Энкодер для {col} не найден")
        
        return df_encoded
    
    def prepare_features(self, df, target_col='price'):
        """Подготовка признаков для обучения"""
        print("\n" + "="*60)
        print("ПОДГОТОВКА ПРИЗНАКОВ")
        print("="*60)
        
        # Определяем признаки для обучения
        exclude_cols = [target_col] + self.categorical_cols
        self.feature_columns = [col for col in df.columns 
                               if col not in exclude_cols 
                               and not col.startswith('Unnamed')]
        
        # Добавляем полиномиальные признаки для улучшения модели
        print(f"\nБазовые признаки ({len(self.feature_columns)}): {self.feature_columns}")
        
        # Создаем DataFrame с признаками
        X = df[self.feature_columns].copy()
        y = df[target_col].copy()
        
        # Добавляем взаимодействия признаков
        # Например, объем бриллианта
        if all(col in X.columns for col in ['x', 'y', 'z']):
            X['volume'] = X['x'] * X['y'] * X['z']
            X['volume'].replace(0, X['volume'].median(), inplace=True)
            print("✓ Добавлен признак: volume (объем)")
        
        # Добавляем соотношения сторон
        if all(col in X.columns for col in ['x', 'y']):
            X['xy_ratio'] = X['x'] / (X['y'] + 1e-10)
            print("✓ Добавлен признак: xy_ratio")
        
        # Логарифмические трансформации
        X['carat_log'] = np.log1p(X['carat'])
        print("✓ Добавлен признак: carat_log")
        
        self.feature_columns = list(X.columns)
        print(f"\nВсего признаков после инжиниринга: {len(self.feature_columns)}")
        
        return X, y
    
    def create_preprocessing_pipeline(self):
        """Создание пайплайна предобработки данных"""
        # Разделяем признаки на числовые и закодированные категориальные
        numeric_features = [col for col in self.feature_columns 
                           if not col.endswith('_encoded')]
        
        # Создаем пайплайн предобработки
        numeric_transformer = Pipeline(steps=[
            ('minmax', MinMaxScaler()),
            ('standard', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'  # Оставляем закодированные категории без изменений
        )
        
        print("\n✓ Создан пайплайн предобработки")
        return self.preprocessor
    
    def train_and_evaluate(self, X, y):
        """Обучение и оценка модели"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛИ")
        print("="*60)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"\nРазмер выборок:")
        print(f"Обучающая: {X_train.shape}")
        print(f"Тестовая: {X_test.shape}")
        
        # Предобработка данных
        preprocessor = self.create_preprocessing_pipeline()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Пробуем разные модели
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1, max_iter=10000)
        }
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models.items():
            print(f"\n{'-'*40}")
            print(f"Модель: {name}")
            
            # Обучение
            model.fit(X_train_processed, y_train)
            
            # Предсказания
            y_train_pred = model.predict(X_train_processed)
            y_test_pred = model.predict(X_test_processed)
            
            # Метрики
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            print(f"R² Train: {train_r2:.4f}, Test: {test_r2:.4f}")
            print(f"RMSE Train: ${train_rmse:,.2f}, Test: ${test_rmse:,.2f}")
            print(f"MAE Train: ${train_mae:,.2f}, Test: ${test_mae:,.2f}")
            
            # Кросс-валидация
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_processed, y_train, 
                                       cv=cv, scoring='r2')
            print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            if test_r2 > best_score:
                best_score = test_r2
                best_model = model
                best_model_name = name
        
        print(f"\n{'='*60}")
        print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
        print(f"Тестовый R²: {best_score:.4f}")
        
        self.model = best_model
        return best_model
    
    def analyze_feature_importance(self, feature_names):
        """Анализ важности признаков"""
        print("\n" + "="*60)
        print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("="*60)
        
        if hasattr(self.model, 'coef_'):
            # Получаем коэффициенты
            coefs = self.model.coef_
            
            # Создаем DataFrame с важностью признаков
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefs,
                'abs_coefficient': np.abs(coefs)
            })
            importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            
            print("\nТоп-10 наиболее важных признаков:")
            for idx, row in importance_df.head(10).iterrows():
                sign = "+" if row['coefficient'] >= 0 else "-"
                print(f"{row['feature']:20}: {sign} {abs(row['coefficient']):.4f}")
            
            return importance_df
    
    def train_final_model(self, X, y):
        """Обучение финальной модели на всех данных"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
        print("="*60)
        
        # Предобработка всех данных
        X_processed = self.preprocessor.fit_transform(X)
        
        # Обучение модели на всех данных
        self.model.fit(X_processed, y)
        
        # Оценка на обучающих данных
        y_pred = self.model.predict(X_processed)
        final_r2 = r2_score(y, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred))
        final_mae = mean_absolute_error(y, y_pred)
        
        print(f"\nФинальная модель:")
        print(f"R²: {final_r2:.4f}")
        print(f"RMSE: ${final_rmse:,.2f}")
        print(f"MAE: ${final_mae:,.2f}")
        
        return self.model
    
    def save_model(self, directory='model_artifacts'):
        """Сохранение модели и всех компонентов"""
        print("\n" + "="*60)
        print("СОХРАНЕНИЕ МОДЕЛИ")
        print("="*60)
        
        # Создаем директорию если её нет
        os.makedirs(directory, exist_ok=True)
        
        # Сохраняем компоненты
        joblib.dump(self.model, f'{directory}/final_model.pkl')
        joblib.dump(self.preprocessor, f'{directory}/preprocessor.pkl')
        joblib.dump(self.label_encoders, f'{directory}/label_encoders.pkl')
        joblib.dump(self.feature_columns, f'{directory}/feature_columns.pkl')
        
        print(f"✓ Модель и компоненты сохранены в директорию '{directory}/'")
    
    def load_model(self, directory='model_artifacts'):
        """Загрузка сохраненной модели"""
        self.model = joblib.load(f'{directory}/final_model.pkl')
        self.preprocessor = joblib.load(f'{directory}/preprocessor.pkl')
        self.label_encoders = joblib.load(f'{directory}/label_encoders.pkl')
        self.feature_columns = joblib.load(f'{directory}/feature_columns.pkl')
        print(f"✓ Модель загружена из директории '{directory}/'")
    
    def predict_test(self, test_filepath, output_file='submission.csv'):
        """Предсказание на тестовых данных"""
        print("\n" + "="*60)
        print("ПРЕДСКАЗАНИЕ НА ТЕСТОВЫХ ДАННЫХ")
        print("="*60)
        
        # Загрузка тестовых данных
        df_test = pd.read_csv(test_filepath)
        test_ids = df_test['id'].copy()
        
        print(f"Тестовых образцов: {len(df_test)}")
        
        # Кодирование категориальных переменных
        df_test_encoded = self.encode_categorical(df_test, fit=False)
        
        # Подготовка признаков
        X_test = df_test_encoded[self.feature_columns].copy()
        
        # Добавляем те же признаки что и при обучении
        if 'x' in X_test.columns and 'y' in X_test.columns and 'z' in X_test.columns:
            X_test['volume'] = X_test['x'] * X_test['y'] * X_test['z']
            X_test['volume'].replace(0, X_test['volume'].median(), inplace=True)
        
        if 'x' in X_test.columns and 'y' in X_test.columns:
            X_test['xy_ratio'] = X_test['x'] / (X_test['y'] + 1e-10)
        
        if 'carat' in X_test.columns:
            X_test['carat_log'] = np.log1p(X_test['carat'])
        
        # Используем только признаки, которые были при обучении
        X_test = X_test[[col for col in self.feature_columns if col in X_test.columns]]
        
        # Предобработка
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Предсказание
        predictions = self.model.predict(X_test_processed)
        
        # Проверка и корректировка предсказаний
        predictions = np.maximum(predictions, 0)  # Убираем отрицательные цены
        
        print(f"\nСтатистика предсказаний:")
        print(f"Минимум: ${predictions.min():.2f}")
        print(f"Максимум: ${predictions.max():.2f}")
        print(f"Среднее: ${predictions.mean():.2f}")
        print(f"Медиана: ${np.median(predictions):.2f}")
        
        # Сохранение результатов
        df_result = pd.DataFrame({
            'id': test_ids,
            'price': predictions.round(2)  # Округляем до 2 знаков
        })
        
        df_result.to_csv(output_file, index=False)
        print(f"\n✓ Результаты сохранены в '{output_file}'")
        
        return df_result

def main():
    """Основная функция выполнения"""
    print("="*60)
    print("ПРОГНОЗИРОВАНИЕ ЦЕНЫ АЛМАЗОВ")
    print("="*60)
    
    # Создаем экземпляр класса
    predictor = DiamondPricePredictor()
    
    # 1. Загрузка и анализ данных
    df_train = predictor.load_and_explore_data('diamonds_train.csv')
    
    # 2. Проверка пропущенных значений
    predictor.check_missing_values(df_train)
    
    # 3. Анализ категориальных переменных
    predictor.analyze_categorical(df_train)
    
    # 4. Кодирование категориальных переменных
    df_train_encoded = predictor.encode_categorical(df_train, fit=True)
    
    # 5. Подготовка признаков
    X, y = predictor.prepare_features(df_train_encoded)
    
    # 6. Обучение и оценка модели
    best_model = predictor.train_and_evaluate(X, y)
    
    # 7. Анализ важности признаков
    feature_names = predictor.preprocessor.get_feature_names_out()
    importance_df = predictor.analyze_feature_importance(feature_names)
    
    # 8. Обучение финальной модели
    predictor.train_final_model(X, y)
    
    # 9. Сохранение модели
    predictor.save_model()
    
    # 10. Предсказание на тестовых данных
    predictions = predictor.predict_test('diamonds_test.csv')
    
    print("\n" + "="*60)
    print("ВЫПОЛНЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
    print("="*60)

if __name__ == "__main__":
    main()
