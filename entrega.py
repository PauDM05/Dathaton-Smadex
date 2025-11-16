import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb

train_root = '/kaggle/input/smadex-challenge-predict-the-revenue/train/train/*/*.parquet'
test_root = '/kaggle/input/smadex-challenge-predict-the-revenue/test/test/*/*.parquet'
dask.config.set({"dataframe.convert-string": False})
ddf_train = dd.read_parquet(train_root)
ddf_test = dd.read_parquet(test_root)
target = 'iap_revenue_d7'

cat_cols = [
    'advertiser_bundle', 'advertiser_category', 'advertiser_subcategory',
    'country', 'region', 'dev_make', 'dev_model', 'dev_os', 'dev_osv',
    'hour', 'weekday'
]
base_num_cols = [
    'avg_act_days', 'avg_daily_sessions', 'avg_days_ins', 'avg_duration',
    'weeks_since_first_seen', 'wifi_ratio', 'weekend_ratio', 'release_msrp'
]
hist_num_candidates = [
    'iap_revenue_usd_bundle', 'iap_revenue_usd_category', 'num_buys_bundle',
    'num_buys_category', 'whale_users_bundle_total_revenue', 'whale_users_bundle_total_num_buys',
    'cpm', 'ctr'
]
num_cols = base_num_cols + hist_num_candidates
existing_cat = [c for c in cat_cols if c in ddf_train.columns]
existing_num = [c for c in num_cols if c in ddf_train.columns]
simple_features = existing_cat + existing_num

ddf_train_simple = ddf_train[simple_features + [target]]
# Aquí incluimos 'row_id' junto con las features en test
# para tener disponible el identificador para el submission
simple_features_test = simple_features + ['row_id']
ddf_test_simple = ddf_test[simple_features_test]

for col in existing_cat:
    ddf_train_simple[col] = ddf_train_simple[col].astype('category')
    ddf_test_simple[col] = ddf_test_simple[col].astype('category')
for col in existing_num:
    ddf_train_simple[col] = ddf_train_simple[col].map_partitions(lambda x: pd.to_numeric(x, errors='coerce'))
    ddf_test_simple[col] = ddf_test_simple[col].map_partitions(lambda x: pd.to_numeric(x, errors='coerce'))

sample_frac = 0.15
train_sample = ddf_train_simple.sample(frac=sample_frac, random_state=42).compute()

valid_num_cols = []
for col in existing_num:
    na_ratio = train_sample[col].isna().mean()
    if na_ratio < 0.98:
        valid_num_cols.append(col)
existing_num = valid_num_cols
simple_features = existing_cat + existing_num
for col in existing_num:
    if ('num_buys' in col) or ('revenue' in col) or (col in ['cpm', 'ctr']):
        train_sample[col] = train_sample[col].fillna(0.0)
train_sample[target] = train_sample[target].clip(lower=0)
train_sample['target_log'] = np.log1p(train_sample[target])

X = train_sample[simple_features]
y = train_sample['target_log']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

reg = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    random_state=42,
    n_jobs=-1
)
reg.fit(
    X_train, y_train,
    categorical_feature=existing_cat,
    eval_set=[(X_valid, y_valid)],
    eval_metric='rmse',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
    ]
)

y_pred_valid_log = reg.predict(X_valid)
y_pred_valid = np.expm1(y_pred_valid_log)
y_true_valid = np.expm1(y_valid)
y_true_valid = np.clip(y_true_valid, 0, None)
y_pred_valid = np.clip(y_pred_valid, 0, None)
msle = mean_squared_log_error(y_true_valid, y_pred_valid)
print(f"MSLE valid: {msle}")
print(f"RMSLE valid: {np.sqrt(msle)}")

# Para test, computamos el DataFrame incluyendo 'row_id'
test_sample = ddf_test_simple.compute()
for col in existing_num:
    if ('num_buys' in col) or ('revenue' in col) or (col in ['cpm', 'ctr']):
        test_sample[col] = test_sample[col].fillna(0.0)
y_test_pred_log = reg.predict(test_sample[simple_features])
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.clip(y_test_pred, 0, None)

submission = pd.DataFrame({
    'row_id': test_sample['row_id'],
    'iap_revenue_d7': y_test_pred
})
submission.to_csv('submission.csv', index=False)

test_root = '/kaggle/input/smadex-challenge-predict-the-revenue/test/test/*/*.parquet'
ddf_test = dd.read_parquet(test_root)


ddf_test_simple = ddf_test[simple_features + ['row_id']]


for col in existing_cat:
    ddf_test_simple[col] = ddf_test_simple[col].astype('category')
for col in existing_num:
    ddf_test_simple[col] = ddf_test_simple[col].map_partitions(
        lambda x: pd.to_numeric(x, errors='coerce')
    )


test_sample = ddf_test_simple.compute()


for col in existing_num:
    if ('num_buys' in col) or ('revenue' in col) or (col in ['cpm', 'ctr']):
        test_sample[col] = test_sample[col].fillna(0.0)


y_test_pred_log = reg.predict(test_sample[simple_features])
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.clip(y_test_pred, 0, None)


submission = pd.DataFrame({
    'row_id': test_sample['row_id'],
    'iap_revenue_d7': y_test_pred
})
submission.to_csv('submission.csv', index=False)

print(submission.head())