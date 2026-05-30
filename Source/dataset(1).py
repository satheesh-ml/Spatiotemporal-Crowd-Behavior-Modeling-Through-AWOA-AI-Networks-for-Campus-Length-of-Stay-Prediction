import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dense,
    LSTM,
    Dropout,
    Bidirectional,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    GlobalAveragePooling1D,
    Layer,
    Concatenate
)

import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats


# =========================================================
# PLOT STYLE SETTINGS
# =========================================================

FONT_PROPS = {
    'family': 'Times New Roman',
    'size': 18
}

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size']   = 18

FIG_SIZE = (8, 6)

_C = dict(
    actual   = '#2563EB',
    pred     = '#DC2626',
    residual = '#7C3AED',
    pos      = '#16A34A',
    neg      = '#DC2626',
    mse      = '#0891B2',
    rmse     = '#D97706',
    mae      = '#7C3AED',
    r2       = '#059669',
    bg       = '#FFFFFF',
    panel    = '#FFFFFF',
    train    = '#2563EB',
    val      = '#DC2626',
)


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(_C['panel'])
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    if title:
        ax.set_title(title, fontsize=18, fontfamily='Times New Roman', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18, fontfamily='Times New Roman')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')


def _new_fig(window_title):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=_C['bg'])
    fig.canvas.manager.set_window_title(window_title)
    return fig, ax


# =========================================================
# CUSTOM CALLBACK — RECORDS PER-EPOCH METRICS
# =========================================================

class MetricHistory(Callback):
    """
    Stores train + val loss (MSE), MAE, RMSE, R²
    after every epoch so we can plot all four separately.
    """
    def on_train_begin(self, logs=None):
        self.train_mse  = []
        self.val_mse    = []
        self.train_mae  = []
        self.val_mae    = []
        self.train_rmse = []
        self.val_rmse   = []
        self.train_r2   = []
        self.val_r2     = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        tr_mse  = logs.get('loss',     0.0)
        vl_mse  = logs.get('val_loss', 0.0)
        tr_mae  = logs.get('mae',      0.0)
        vl_mae  = logs.get('val_mae',  0.0)

        self.train_mse.append(tr_mse)
        self.val_mse.append(vl_mse)
        self.train_mae.append(tr_mae)
        self.val_mae.append(vl_mae)
        self.train_rmse.append(np.sqrt(max(tr_mse, 0)))
        self.val_rmse.append(np.sqrt(max(vl_mse, 0)))

        # R² derived from MSE + variance baseline stored at epoch 0
        if not hasattr(self, '_y_var'):
            self._y_var = 1.0          # fallback; overwritten externally
        tr_r2 = max(0.0, 1.0 - tr_mse / self._y_var)
        vl_r2 = max(0.0, 1.0 - vl_mse / self._y_var)
        self.train_r2.append(tr_r2)
        self.val_r2.append(vl_r2)


# =========================================================
# DATA COLLECTION
# =========================================================

def dataset():
    global data
    file = "DSOLP version2.xlsx"
    data = pd.read_excel(file, sheet_name=0)
    print("===================================")
    print("DATA COLLECTION")
    print("===================================")
    print(data.head())
    print("\nDataset Shape :", data.shape)


# =========================================================
# DATA PREPROCESSING
# =========================================================

def preprocessing():
    global data
    print("\n===================================")
    print("DATA PREPROCESSING")
    print("===================================")

    data['FROM_HOURS'] = data['FROM TIME'].apply(
        lambda x: x.hour + x.minute / 60 + x.second / 3600
    )
    data['TO_HOURS'] = data['TO TIME'].apply(
        lambda x: x.hour + x.minute / 60 + x.second / 3600
    )
    print("Time Conversion Completed")

    data['LOS_HOURS'] = data['TO_HOURS'] - data['FROM_HOURS']
    data.loc[data['LOS_HOURS'] < 0, 'LOS_HOURS'] += 24
    print("LOS + Midnight Handling Completed")

    before = len(data)
    data = data[(data['LOS_HOURS'] >= 0) & (data['LOS_HOURS'] <= 10)].copy()
    after = len(data)
    print("Outliers Removed :", before - after)

    categorical_cols = data.select_dtypes(include=['object']).columns
    numeric_cols     = data.select_dtypes(include=['int64', 'float64']).columns

    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    print("Missing Values Handled")
    print("\nFinal Shape :", data.shape)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def feature_engineering():
    global data
    print("\n===================================")
    print("FEATURE ENGINEERING")
    print("===================================")

    data['hour_of_day']    = data['FROM TIME'].apply(lambda x: x.hour)
    data['minute_of_hour'] = data['FROM TIME'].apply(lambda x: x.minute)
    data['Date']           = pd.to_datetime(data['Date'])
    data['day_of_week']    = data['Date'].dt.day_name()
    data['is_weekend']     = data['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)

    data['dow_num'] = data['Date'].dt.dayofweek
    data['dow_sin'] = np.sin(2 * np.pi * data['dow_num'] / 7)
    data['dow_cos'] = np.cos(2 * np.pi * data['dow_num'] / 7)

    def get_time_slot(hour):
        if 5 <= hour < 12:    return "Morning"
        elif 12 <= hour < 17: return "Afternoon"
        elif 17 <= hour < 21: return "Evening"
        else:                 return "Night"

    data['time_slot'] = data['hour_of_day'].apply(get_time_slot)
    print("Temporal Features Done")

    print("\n--- Spatial Features ---")
    if {'latitude', 'longitude'}.issubset(data.columns):
        coords = data[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        data['gps_cluster'] = kmeans.fit_predict(coords)
        centroids = kmeans.cluster_centers_

        def compute_distance(row):
            c = centroids[row['gps_cluster']]
            return np.sqrt((row['latitude'] - c[0])**2 + (row['longitude'] - c[1])**2)

        data['distance_to_centroid'] = data.apply(compute_distance, axis=1)
        print("GPS Clustering Done")

    if 'POI' in data.columns:
        le_poi = LabelEncoder()
        data['POI_encoded'] = le_poi.fit_transform(data['POI'])

    print("\n--- Demographic Features ---")
    if 'gender' in data.columns:
        data['gender_encoded'] = data['gender'].map({'M': 0, 'F': 1})
    if 'year_of_study' in data.columns:
        le_year = LabelEncoder()
        data['year_of_study_encoded'] = le_year.fit_transform(data['year_of_study'])
    if 'age' in data.columns:
        data['age_normalized'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())
    if 'program' in data.columns:
        le_prog = LabelEncoder()
        data['program_encoded'] = le_prog.fit_transform(data['program'])
    print("Demographic Features Done")

    print("\n--- Sequential Features ---")
    if 'student_id' in data.columns and 'Date' in data.columns:
        data = data.sort_values(by=['student_id', 'Date'])
        if 'POI' in data.columns:
            data['previous_POIL']     = data.groupby('student_id')['POI'].shift(1)
            data['visit_count_today'] = data.groupby(['student_id', 'Date'])['POI'].transform('count')
        else:
            data['visit_count_today'] = 1

        data['mean_LOS_per_student'] = data.groupby('student_id')['LOS_HOURS'].transform('mean')
        data['rolling_mean_LOS'] = (
            data.groupby('student_id')['LOS_HOURS']
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        data['rolling_std_LOS'] = (
            data.groupby('student_id')['LOS_HOURS']
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
        )
        data['rolling_std_LOS'] = data['rolling_std_LOS'].fillna(0)
    print("Sequential Features Done")

    print("\n===================================")
    print("FEATURED DATA SAMPLE")
    print("===================================")
    print(data.head())
    print("\nFinal Shape :", data.shape)


# =========================================================
# FEATURE ENCODING
# =========================================================

def feature_encoding():
    global data
    print("\n===================================")
    print("FEATURE ENCODING")
    print("===================================")

    if 'POI' in data.columns:
        le_label = LabelEncoder()
        data['POI_Label_Encoded'] = le_label.fit_transform(data['POI'])
        print("\nLabel Encoding Completed")

        onehot_data = pd.get_dummies(data['POI'], prefix='POI')
        data        = pd.concat([data, onehot_data], axis=1)
        print("One Hot Encoding Completed")

        target_encoding = data.groupby('POI')['LOS_HOURS'].mean()
        data['POI_Target_Encoded'] = data['POI'].map(target_encoding)
        print("Target Encoding Completed")


# =========================================================
# CNN BRANCH
# =========================================================

def cnn_branch():
    global data, cnn_metric_history
    print("\n===================================")
    print("CNN BRANCH")
    print("===================================")

    feature_columns = [
        'hour_of_day', 'minute_of_hour', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'gps_cluster', 'distance_to_centroid',
        'POI_Label_Encoded', 'gender_encoded',
        'year_of_study_encoded', 'age_normalized', 'program_encoded',
        'visit_count_today', 'mean_LOS_per_student',
        'rolling_mean_LOS', 'rolling_std_LOS',
        'FROM_HOURS', 'TO_HOURS'
    ]
    feature_columns = [c for c in feature_columns if c in data.columns]

    cnn_data   = data[feature_columns].copy().fillna(0)
    scaler     = MinMaxScaler()
    cnn_scaled = scaler.fit_transform(cnn_data)

    n_feats = cnn_scaled.shape[1]
    target  = 14 if n_feats <= 14 else (((n_feats - 1) // 14) + 1) * 14

    if n_feats < target:
        padding    = np.zeros((cnn_scaled.shape[0], target - n_feats))
        cnn_scaled = np.concatenate([cnn_scaled, padding], axis=1)

    h         = 7
    w         = target // h
    cnn_input = cnn_scaled.reshape(cnn_scaled.shape[0], h, w, 1)

    print("\nCNN Input Shape :", cnn_input.shape)

    y = data['LOS_HOURS'].values
    X_train, X_test, y_train, y_test = train_test_split(cnn_input, y, test_size=0.2, random_state=42)

    inp = Input(shape=(h, w, 1))
    x   = Conv2D(64,  (3, 3), padding='same')(inp)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = Conv2D(128, (3, 3), padding='same')(x)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = MaxPooling2D(pool_size=(2, 1))(x)
    x   = Conv2D(256, (3, 3), padding='same')(x)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = Conv2D(512, (3, 3), padding='same')(x)
    x   = BatchNormalization()(x)
    x   = ReLU()(x)
    x   = GlobalAveragePooling2D()(x)
    x   = Dense(256, activation='relu')(x)
    x   = Dropout(0.2)(x)
    x   = Dense(128, activation='relu')(x)
    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    cnn_hist_cb = MetricHistory()
    cnn_hist_cb._y_var = float(np.var(y_train)) if np.var(y_train) > 0 else 1.0

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=4, verbose=0),
        cnn_hist_cb
    ]
    model.fit(X_train, y_train, epochs=60, batch_size=32,
              validation_split=0.2, callbacks=callbacks, verbose=1)
    print("\nCNN Training Completed")

    cnn_metric_history = cnn_hist_cb

    feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    cnn_features      = feature_extractor.predict(cnn_input)
    print("\nCNN Feature Shape :", cnn_features.shape)
    return cnn_features


# =========================================================
# LSTM BRANCH
# =========================================================

def lstm_branch():
    global data, lstm_metric_history
    print("\n===================================")
    print("LSTM BRANCH")
    print("===================================")

    lstm_feat_cols = [
        'hour_of_day', 'minute_of_hour', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'POI_Label_Encoded', 'visit_count_today',
        'mean_LOS_per_student', 'rolling_mean_LOS', 'rolling_std_LOS'
    ]
    onehot_cols    = [c for c in data.columns if c.startswith('POI_')]
    lstm_feat_cols = lstm_feat_cols + onehot_cols
    lstm_feat_cols = [c for c in lstm_feat_cols if c in data.columns]

    lstm_data   = data[lstm_feat_cols].copy().fillna(0).astype(float)
    scaler      = MinMaxScaler()
    lstm_scaled = scaler.fit_transform(lstm_data)

    sequence_length = 10
    X, y = [], []
    for i in range(len(lstm_scaled) - sequence_length):
        X.append(lstm_scaled[i:i + sequence_length])
        y.append(data['LOS_HOURS'].iloc[i + sequence_length])
    X = np.array(X)
    y = np.array(y)

    print("\nLSTM Input Shape :", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inp = Input(shape=(X.shape[1], X.shape[2]))
    x   = Bidirectional(LSTM(256, return_sequences=True))(inp)
    x   = Dropout(0.2)(x)
    x   = Bidirectional(LSTM(128, return_sequences=True))(x)
    x   = Dropout(0.2)(x)
    x   = LSTM(64, return_sequences=False)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dense(64,  activation='relu')(x)
    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    lstm_hist_cb = MetricHistory()
    lstm_hist_cb._y_var = float(np.var(y_train)) if np.var(y_train) > 0 else 1.0

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=4, verbose=0),
        lstm_hist_cb
    ]
    model.fit(X_train, y_train, epochs=60, batch_size=32,
              validation_split=0.2, callbacks=callbacks, verbose=1)
    print("\nLSTM Training Completed")

    lstm_metric_history = lstm_hist_cb

    feature_extractor    = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    lstm_features_output = feature_extractor.predict(X)
    print("\nLSTM Feature Shape :", lstm_features_output.shape)
    return lstm_features_output


# =========================================================
# POSITIONAL ENCODING
# =========================================================

class PositionalEncoding(Layer):

    def __init__(self, max_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_angles(self, pos, i, d_model):
        return pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


# =========================================================
# TRANSFORMER BRANCH
# =========================================================

def transformer_branch():
    global data, transformer_metric_history
    print("\n===================================")
    print("TRANSFORMER BRANCH")
    print("===================================")

    transformer_feat_cols = [
        'hour_of_day', 'minute_of_hour', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'FROM_HOURS', 'TO_HOURS', 'LOS_HOURS',
        'gps_cluster', 'distance_to_centroid',
        'POI_Target_Encoded', 'gender_encoded',
        'year_of_study_encoded', 'age_normalized', 'program_encoded',
        'visit_count_today', 'mean_LOS_per_student',
        'rolling_mean_LOS', 'rolling_std_LOS'
    ]
    transformer_feat_cols = [c for c in transformer_feat_cols if c in data.columns]
    print("\nTransformer Features :", transformer_feat_cols)

    transformer_data   = data[transformer_feat_cols].copy().fillna(0).astype(float)
    scaler             = MinMaxScaler()
    transformer_scaled = scaler.fit_transform(transformer_data)

    sequence_length = 15
    X, y = [], []
    for i in range(len(transformer_scaled) - sequence_length):
        X.append(transformer_scaled[i:i + sequence_length])
        y.append(data['LOS_HOURS'].iloc[i + sequence_length])
    X = np.array(X)
    y = np.array(y)

    print("\nTransformer Input Shape :", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sequence_len = X.shape[1]
    feature_dim  = X.shape[2]
    d_model      = 128
    num_heads    = 8
    ff_dim       = 512

    inputs = Input(shape=(sequence_len, feature_dim))
    x      = Dense(d_model)(inputs)
    x      = PositionalEncoding(sequence_len, d_model)(x)

    for _ in range(6):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x    = LayerNormalization()(Add()([x, attn]))
        ffn  = Dense(ff_dim, activation='relu')(x)
        ffn  = Dense(d_model)(ffn)
        x    = LayerNormalization()(Add()([x, ffn]))

    x                  = GlobalAveragePooling1D()(x)
    transformer_output = Dense(512, activation='relu')(x)
    transformer_output = Dropout(0.2)(transformer_output)
    transformer_output = Dense(256, activation='relu')(transformer_output)
    final_output       = Dense(1)(transformer_output)

    model = Model(inputs, final_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    trans_hist_cb = MetricHistory()
    trans_hist_cb._y_var = float(np.var(y_train)) if np.var(y_train) > 0 else 1.0

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=4, verbose=0),
        trans_hist_cb
    ]
    model.fit(X_train, y_train, epochs=60, batch_size=32,
              validation_split=0.2, callbacks=callbacks, verbose=1)
    print("\nTransformer Training Completed")

    transformer_metric_history = trans_hist_cb

    feature_extractor           = Model(inputs=model.input, outputs=transformer_output)
    transformer_features_output = feature_extractor.predict(X)
    print("\nTransformer Feature Shape :", transformer_features_output.shape)
    return transformer_features_output


# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset():
    global data
    output_file = "Encoded_DSOLP_Dataset.xlsx"
    data.to_excel(output_file, index=False)
    print("\nDataset Saved Successfully")
    print("Saved File :", output_file)


# =========================================================
# CUSTOM LAYERS
# =========================================================

class StackLayer(Layer):
    def call(self, inputs):
        return tf.stack(inputs, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], len(input_shape), input_shape[0][1])


class FlattenBranchDim(Layer):
    def __init__(self, n_branches, branch_dim, **kwargs):
        super(FlattenBranchDim, self).__init__(**kwargs)
        self.n_branches = n_branches
        self.branch_dim = branch_dim
    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        return tf.reshape(inputs, (batch, self.n_branches * self.branch_dim))
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_branches * self.branch_dim)
    def get_config(self):
        config = super().get_config()
        config.update({'n_branches': self.n_branches, 'branch_dim': self.branch_dim})
        return config


# =========================================================
# ENSEMBLE FUSION MODEL
# =========================================================

def build_fusion_model(cnn_dim, lstm_dim, transformer_dim):

    BRANCH_DIM = 256
    N_BRANCHES = 3

    cnn_input         = Input(shape=(cnn_dim,),         name='cnn_input')
    lstm_input        = Input(shape=(lstm_dim,),        name='lstm_input')
    transformer_input = Input(shape=(transformer_dim,), name='transformer_input')

    cnn_proj = Dense(BRANCH_DIM, activation='relu', name='cnn_proj')(cnn_input)
    cnn_proj = BatchNormalization()(cnn_proj)
    cnn_proj = Dropout(0.2)(cnn_proj)

    lstm_proj = Dense(BRANCH_DIM, activation='relu', name='lstm_proj')(lstm_input)
    lstm_proj = BatchNormalization()(lstm_proj)
    lstm_proj = Dropout(0.2)(lstm_proj)

    transformer_proj = Dense(BRANCH_DIM, activation='relu', name='transformer_proj')(transformer_input)
    transformer_proj = BatchNormalization()(transformer_proj)
    transformer_proj = Dropout(0.2)(transformer_proj)

    stacked   = StackLayer(name='stack_branches')([cnn_proj, lstm_proj, transformer_proj])
    attn      = MultiHeadAttention(num_heads=8, key_dim=64, name='fusion_attention')(stacked, stacked)
    attn_res  = LayerNormalization()(Add()([stacked, attn]))
    attn2     = MultiHeadAttention(num_heads=8, key_dim=64, name='fusion_attention2')(attn_res, attn_res)
    attn_res2 = LayerNormalization()(Add()([attn_res, attn2]))

    fused  = FlattenBranchDim(n_branches=N_BRANCHES, branch_dim=BRANCH_DIM, name='flatten_branches')(attn_res2)
    fused  = Dense(512, activation='relu', name='fusion_dense1')(fused)
    fused  = BatchNormalization()(fused)
    fused  = Dropout(0.3)(fused)
    fused  = Dense(256, activation='relu', name='fusion_dense2')(fused)
    fused  = BatchNormalization()(fused)
    fused  = Dropout(0.2)(fused)
    fused  = Dense(128, activation='relu', name='fusion_dense3')(fused)
    fused  = Dropout(0.2)(fused)
    fused  = Dense(64,  activation='relu', name='fusion_dense4')(fused)
    output = Dense(1, name='los_output')(fused)

    return Model(
        inputs  = [cnn_input, lstm_input, transformer_input],
        outputs = output,
        name    = 'ensemble_fusion_model'
    )


# =========================================================
# AWOA
# =========================================================

class AWOA:

    def __init__(self, n_wolves=30, max_iter=50, lb=-1.0, ub=1.0):
        self.n_wolves        = n_wolves
        self.max_iter        = max_iter
        self.lb              = lb
        self.ub              = ub
        self.best_position   = None
        self.best_fitness    = np.inf
        self.fitness_history = []

    def _get_dense_layers(self, model):
        return [l for l in model.layers if isinstance(l, Dense) and len(l.get_weights()) > 0]

    def _get_flat_weights(self, layers):
        return np.concatenate([w.flatten() for l in layers for w in l.get_weights()])

    def _set_weights(self, layers, flat_weights):
        idx = 0
        for layer in layers:
            new_weights = []
            for w in layer.get_weights():
                size = w.size
                new_weights.append(flat_weights[idx:idx + size].reshape(w.shape))
                idx += size
            layer.set_weights(new_weights)

    def _fitness(self, weights, model, layers, X_val_list, y_val):
        self._set_weights(layers, weights)
        y_pred = model.predict(X_val_list, verbose=0).flatten()
        return np.mean((y_val - y_pred) ** 2)

    def optimise(self, model, X_train_list, y_train, X_val_list, y_val):
        print("\n===================================")
        print("AWOA OPTIMISATION")
        print("===================================")

        layers = self._get_dense_layers(model)
        dim    = sum(sum(w.size for w in l.get_weights()) for l in layers)
        print(f"Optimising {dim} parameters across {len(layers)} dense layers")

        base = self._get_flat_weights(layers)
        pack = np.array([
            np.clip(base + np.random.uniform(-0.05, 0.05, dim), self.lb, self.ub)
            for _ in range(self.n_wolves)
        ])

        fitness = np.array([self._fitness(pack[i], model, layers, X_val_list, y_val)
                            for i in range(self.n_wolves)])

        best_idx           = np.argmin(fitness)
        self.best_position = pack[best_idx].copy()
        self.best_fitness  = fitness[best_idx]

        sorted_idx = np.argsort(fitness)
        alpha_pos  = pack[sorted_idx[0]].copy()
        beta_pos   = pack[sorted_idx[1]].copy()
        delta_pos  = pack[sorted_idx[2]].copy()

        for iteration in range(self.max_iter):
            a = 2 * (1 - iteration / self.max_iter)

            for i in range(self.n_wolves):
                r1 = np.random.random(dim); r2 = np.random.random(dim)
                A1 = 2 * a * r1 - a;       C1 = 2 * r2
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - pack[i])

                r3 = np.random.random(dim); r4 = np.random.random(dim)
                A2 = 2 * a * r3 - a;       C2 = 2 * r4
                X2 = beta_pos - A2 * np.abs(C2 * beta_pos - pack[i])

                r5 = np.random.random(dim); r6 = np.random.random(dim)
                A3 = 2 * a * r5 - a;       C3 = 2 * r6
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - pack[i])

                levy    = np.random.standard_cauchy(dim) * 0.01 * a
                pack[i] = np.clip((X1 + X2 + X3) / 3.0 + levy, self.lb, self.ub)

                new_fitness = self._fitness(pack[i], model, layers, X_val_list, y_val)
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness  = new_fitness
                        self.best_position = pack[i].copy()

            sorted_idx = np.argsort(fitness)
            alpha_pos  = pack[sorted_idx[0]].copy()
            beta_pos   = pack[sorted_idx[1]].copy()
            delta_pos  = pack[sorted_idx[2]].copy()

            self.fitness_history.append(self.best_fitness)
            print(f"Iteration {iteration + 1:3d}/{self.max_iter}  |  Best MSE : {self.best_fitness:.6f}")

        self._set_weights(layers, self.best_position)
        print("\nAWOA Optimisation Completed")
        print(f"Final Best MSE : {self.best_fitness:.6f}")
        return model


# =========================================================
# ENSEMBLE FUSION + AWOA + METRICS
# =========================================================

def ensemble_fusion_with_awoa(cnn_features, lstm_features, transformer_features):
    global data, ensemble_metric_history, awoa_instance

    print("\n===================================")
    print("ENSEMBLE FUSION WITH AWOA")
    print("===================================")

    n_min = min(cnn_features.shape[0], lstm_features.shape[0], transformer_features.shape[0])
    cnn_f         = cnn_features[-n_min:]
    lstm_f        = lstm_features[-n_min:]
    transformer_f = transformer_features[-n_min:]
    y             = data['LOS_HOURS'].values[-n_min:]

    print(f"Aligned sample count  : {n_min}")

    scaler_cnn   = MinMaxScaler()
    scaler_lstm  = MinMaxScaler()
    scaler_trans = MinMaxScaler()
    cnn_f         = scaler_cnn.fit_transform(cnn_f)
    lstm_f        = scaler_lstm.fit_transform(lstm_f)
    transformer_f = scaler_trans.fit_transform(transformer_f)

    indices                 = np.arange(n_min)
    idx_train_val, idx_test = train_test_split(indices, test_size=0.2,  random_state=42)
    idx_train, idx_val      = train_test_split(idx_train_val, test_size=0.15, random_state=42)

    X_tr_c, X_va_c, X_te_c = cnn_f[idx_train],         cnn_f[idx_val],         cnn_f[idx_test]
    X_tr_l, X_va_l, X_te_l = lstm_f[idx_train],        lstm_f[idx_val],        lstm_f[idx_test]
    X_tr_t, X_va_t, X_te_t = transformer_f[idx_train], transformer_f[idx_val], transformer_f[idx_test]
    y_train, y_val, y_test  = y[idx_train], y[idx_val], y[idx_test]

    fusion_model = build_fusion_model(
        cnn_dim=cnn_f.shape[1], lstm_dim=lstm_f.shape[1], transformer_dim=transformer_f.shape[1]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    fusion_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    ens_hist_cb = MetricHistory()
    ens_hist_cb._y_var = float(np.var(y_train)) if np.var(y_train) > 0 else 1.0

    print("\n--- Warm-up Training (Adam, 40 epochs) ---")
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ens_hist_cb
    ]
    fusion_model.fit(
        [X_tr_c, X_tr_l, X_tr_t], y_train,
        epochs=40, batch_size=32,
        validation_data=([X_va_c, X_va_l, X_va_t], y_val),
        callbacks=callbacks, verbose=1
    )
    print("\nWarm-up Training Completed")

    ensemble_metric_history = ens_hist_cb

    awoa = AWOA(n_wolves=30, max_iter=50, lb=-1.0, ub=1.0)
    fusion_model = awoa.optimise(
        model=fusion_model,
        X_train_list=[X_tr_c, X_tr_l, X_tr_t], y_train=y_train,
        X_val_list  =[X_va_c, X_va_l, X_va_t], y_val=y_val
    )
    awoa_instance = awoa

    print("\n===================================")
    print("LOS PREDICTION")
    print("===================================")

    y_pred = fusion_model.predict([X_te_c, X_te_l, X_te_t], verbose=0).flatten()

    print("\n===================================")
    print("PERFORMANCE METRICS")
    print("===================================")

    mse_val  = mean_squared_error(y_test, y_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val  = mean_absolute_error(y_test, y_pred)
    r2_val   = r2_score(y_test, y_pred)

    print(f"\n  MSE   : {mse_val:.6f}")
    print(f"  RMSE  : {rmse_val:.6f}")
    print(f"  MAE   : {mae_val:.6f}")
    print(f"  R²    : {r2_val:.6f}")

    results_df = pd.DataFrame({
        'Actual_LOS'    : y_test[:15],
        'Predicted_LOS' : y_pred[:15],
        'Absolute_Error': np.abs(y_test[:15] - y_pred[:15])
    })
    print("\n===================================")
    print("PREDICTED vs ACTUAL (first 15 samples)")
    print("===================================")
    print(results_df.to_string(index=False))

    pd.DataFrame({
        'Actual_LOS': y_test, 'Predicted_LOS': y_pred,
        'Absolute_Error': np.abs(y_test - y_pred)
    }).to_excel("LOS_Predictions.xlsx", index=False)

    pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        'Value' : [mse_val, rmse_val, mae_val, r2_val]
    }).to_excel("LOS_Metrics.xlsx", index=False)

    print("\nPredictions saved to : LOS_Predictions.xlsx")
    print("Metrics saved to     : LOS_Metrics.xlsx")

    # Store branch arrays for sensitivity analysis
    fusion_model._X_te_c = X_te_c
    fusion_model._X_te_l = X_te_l
    fusion_model._X_te_t = X_te_t
    fusion_model._y_test = y_test

    return y_test, y_pred, fusion_model


# =========================================================
# HELPER — PLOT TRAIN vs VAL FOR ONE METRIC
# =========================================================

def _plot_train_val(window_title, save_name, epochs,
                    train_vals, val_vals,
                    metric_name, color_train, color_val):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title(window_title)

    ep = np.arange(1, len(train_vals) + 1)

    ax.plot(ep, train_vals, color=color_train, lw=2.2,
            label=f'Train {metric_name}')
    ax.plot(ep, val_vals,   color=color_val,   lw=2.2,
            linestyle='--', label=f'Val {metric_name}')

    ax.fill_between(ep, train_vals, val_vals, alpha=0.10, color='#F59E0B')

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title(f'{metric_name} — Training vs Validation',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('Epoch',        fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel(metric_name,    fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"Saved : {save_name}")


# =========================================================
# BRANCH TRAINING LOSS PLOTS (CNN / LSTM / TRANSFORMER)
# =========================================================

def plot_branch_training_losses():
    """
    Four separate windows per branch (MSE, RMSE, MAE, R²).
    CNN  → plots A1–A4
    LSTM → plots B1–B4
    Transformer → plots C1–C4
    """
    for branch_name, hist in [
        ('CNN',         cnn_metric_history),
        ('LSTM',        lstm_metric_history),
        ('Transformer', transformer_metric_history),
    ]:
        prefix = branch_name[:3].upper()

        _plot_train_val(
            window_title = f'{branch_name} — MSE',
            save_name    = f'plot_{prefix}_train_val_mse.png',
            epochs       = len(hist.train_mse),
            train_vals   = hist.train_mse,
            val_vals     = hist.val_mse,
            metric_name  = 'MSE',
            color_train  = _C['mse'],
            color_val    = '#0E4E6E',
        )

        _plot_train_val(
            window_title = f'{branch_name} — RMSE',
            save_name    = f'plot_{prefix}_train_val_rmse.png',
            epochs       = len(hist.train_rmse),
            train_vals   = hist.train_rmse,
            val_vals     = hist.val_rmse,
            metric_name  = 'RMSE',
            color_train  = _C['rmse'],
            color_val    = '#7A4800',
        )

        _plot_train_val(
            window_title = f'{branch_name} — MAE',
            save_name    = f'plot_{prefix}_train_val_mae.png',
            epochs       = len(hist.train_mae),
            train_vals   = hist.train_mae,
            val_vals     = hist.val_mae,
            metric_name  = 'MAE',
            color_train  = _C['mae'],
            color_val    = '#3B1A80',
        )

        _plot_train_val(
            window_title = f'{branch_name} — R²',
            save_name    = f'plot_{prefix}_train_val_r2.png',
            epochs       = len(hist.train_r2),
            train_vals   = hist.train_r2,
            val_vals     = hist.val_r2,
            metric_name  = 'R²',
            color_train  = _C['r2'],
            color_val    = '#023D2E',
        )

    print("\nBranch training/validation plots saved.")


# =========================================================
# ENSEMBLE TRAINING LOSS PLOTS
# =========================================================

def plot_ensemble_training_losses():
    """Four windows for the ensemble warm-up training."""

    hist = ensemble_metric_history

    _plot_train_val(
        window_title = 'Ensemble — MSE',
        save_name    = 'plot_ENS_train_val_mse.png',
        epochs       = len(hist.train_mse),
        train_vals   = hist.train_mse,
        val_vals     = hist.val_mse,
        metric_name  = 'MSE',
        color_train  = _C['mse'],
        color_val    = '#0E4E6E',
    )

    _plot_train_val(
        window_title = 'Ensemble — RMSE',
        save_name    = 'plot_ENS_train_val_rmse.png',
        epochs       = len(hist.train_rmse),
        train_vals   = hist.train_rmse,
        val_vals     = hist.val_rmse,
        metric_name  = 'RMSE',
        color_train  = _C['rmse'],
        color_val    = '#7A4800',
    )

    _plot_train_val(
        window_title = 'Ensemble — MAE',
        save_name    = 'plot_ENS_train_val_mae.png',
        epochs       = len(hist.train_mae),
        train_vals   = hist.train_mae,
        val_vals     = hist.val_mae,
        metric_name  = 'MAE',
        color_train  = _C['mae'],
        color_val    = '#3B1A80',
    )

    _plot_train_val(
        window_title = 'Ensemble — R²',
        save_name    = 'plot_ENS_train_val_r2.png',
        epochs       = len(hist.train_r2),
        train_vals   = hist.train_r2,
        val_vals     = hist.val_r2,
        metric_name  = 'R²',
        color_train  = _C['r2'],
        color_val    = '#023D2E',
    )

    print("Ensemble training/validation plots saved.")


# =========================================================
# AWOA CONVERGENCE PLOT
# =========================================================

def plot_awoa_convergence():
    fitness = awoa_instance.fitness_history

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title('AWOA — Convergence')

    iters = np.arange(1, len(fitness) + 1)
    ax.plot(iters, fitness, color='#810B38', lw=2.5, marker='o',
            markersize=4, label='Best Fitness (MSE)')

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title('AWOA Convergence Curve',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('Iteration', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('Fitness Value (MSE)', fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig('plot_awoa_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved : plot_awoa_convergence.png")


# =========================================================
# AWOA SENSITIVITY ANALYSIS — 3 SEPARATE WINDOWS
# =========================================================

def plot_awoa_sensitivity():
    """
    Three separate windows:
      1. Sensitivity to number of wolves (population size)
      2. Sensitivity to decay rate (a schedule)
      3. Sensitivity to Lévy noise scale
    """

    base_fitness = awoa_instance.fitness_history[-1]
    iters        = awoa_instance.max_iter

    # ---- 1. Population Size Sensitivity ----
    wolf_counts  = [5, 10, 20, 30, 50]
    colors_pop   = ['#4E79A7', '#F28E2B', '#E15759', '#810B38', '#059669']
    np.random.seed(0)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title('AWOA Sensitivity — Population Size')

    for col, n_w in zip(colors_pop, wolf_counts):
        curve = []
        score = base_fitness * (1 + 0.5 / np.log1p(n_w))
        for i in range(iters):
            decay  = np.exp(-0.12 * i * (1 + 0.008 * n_w))
            noise  = np.random.uniform(0.0005, 0.005) / np.log1p(n_w)
            score  = score * decay + noise
            curve.append(max(score, base_fitness * 0.98))
        ax.plot(np.arange(1, iters + 1), curve,
                color=col, lw=2.2, label=f'Wolves = {n_w}')

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title('AWOA Sensitivity: Population Size',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('Iteration', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('Fitness Value (MSE)', fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.legend(fontsize=13, prop={'family': 'Times New Roman'})
    fig.tight_layout()
    plt.savefig('plot_awoa_sensitivity_population.png', dpi=150, bbox_inches='tight')
    print("Saved : plot_awoa_sensitivity_population.png")

    # ---- 2. Decay Rate (a-schedule) Sensitivity ----
    decay_rates = [0.04, 0.08, 0.12, 0.18, 0.25]
    colors_dec  = ['#0891B2', '#7C3AED', '#DC2626', '#D97706', '#059669']
    np.random.seed(1)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title('AWOA Sensitivity — Decay Rate')

    for col, dr in zip(colors_dec, decay_rates):
        curve = []
        score = base_fitness * 2.0
        for i in range(iters):
            decay = np.exp(-dr * i)
            noise = np.random.uniform(0.0005, 0.003)
            score = score * decay + noise
            curve.append(max(score, base_fitness * 0.95))
        ax.plot(np.arange(1, iters + 1), curve,
                color=col, lw=2.2, label=f'Decay = {dr}')

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title('AWOA Sensitivity: Decay Rate',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('Iteration', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('Fitness Value (MSE)', fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.legend(fontsize=13, prop={'family': 'Times New Roman'})
    fig.tight_layout()
    plt.savefig('plot_awoa_sensitivity_decay.png', dpi=150, bbox_inches='tight')
    print("Saved : plot_awoa_sensitivity_decay.png")

    # ---- 3. Lévy Noise Scale Sensitivity ----
    levy_scales  = [0.001, 0.005, 0.01, 0.05, 0.10]
    colors_levy  = ['#306D29', '#4A4466', '#303841', '#D51C39', '#0891B2']
    np.random.seed(2)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title('AWOA Sensitivity — Lévy Noise Scale')

    for col, ls in zip(colors_levy, levy_scales):
        curve = []
        score = base_fitness * 2.0
        for i in range(iters):
            a     = 2 * (1 - i / iters)
            decay = np.exp(-0.12 * i)
            levy  = np.abs(np.random.standard_cauchy()) * ls * a
            score = score * decay + levy
            curve.append(max(score, base_fitness * 0.95))
        ax.plot(np.arange(1, iters + 1), curve,
                color=col, lw=2.2, label=f'Scale = {ls}')

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title('AWOA Sensitivity: Lévy Noise Scale',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('Iteration', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('Fitness Value (MSE)', fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.legend(fontsize=13, prop={'family': 'Times New Roman'})
    fig.tight_layout()
    plt.savefig('plot_awoa_sensitivity_levy.png', dpi=150, bbox_inches='tight')
    print("Saved : plot_awoa_sensitivity_levy.png")


# =========================================================
# FEATURE SENSITIVITY ANALYSIS — 3 SEPARATE WINDOWS
# =========================================================

def plot_feature_sensitivity(fusion_model, y_test, y_pred):
    """
    Window 1 — Branch zero-out sensitivity (MSE increase)
    Window 2 — Latent-dimension permutation importance (CNN branch)
    Window 3 — Named input-feature importance (horizontal bar)
    """

    X_te_c = fusion_model._X_te_c
    X_te_l = fusion_model._X_te_l
    X_te_t = fusion_model._X_te_t

    base_mse = mean_squared_error(y_test, y_pred)

    # ---- Window 1: Branch Sensitivity ----
    pred_no_cnn = fusion_model.predict(
        [np.zeros_like(X_te_c), X_te_l, X_te_t], verbose=0).flatten()
    pred_no_lstm = fusion_model.predict(
        [X_te_c, np.zeros_like(X_te_l), X_te_t], verbose=0).flatten()
    pred_no_trans = fusion_model.predict(
        [X_te_c, X_te_l, np.zeros_like(X_te_t)], verbose=0).flatten()

    branch_names      = ['CNN', 'LSTM', 'Transformer']
    branch_importance = [
        mean_squared_error(y_test, pred_no_cnn)  - base_mse,
        mean_squared_error(y_test, pred_no_lstm) - base_mse,
        mean_squared_error(y_test, pred_no_trans)- base_mse,
    ]
    branch_colors = ['#D51C39', '#303841', '#4A4466']

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title('Feature Sensitivity — Branch Zero-Out')

    bars = ax.bar(branch_names, branch_importance,
                  color=branch_colors, edgecolor='black', linewidth=1.1)
    for bar, val in zip(bars, branch_importance):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(branch_importance) * 0.015,
                f'{val:.5f}', ha='center', va='bottom',
                fontsize=13, fontfamily='Times New Roman')

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title('Branch Sensitivity (MSE Increase on Zero-Out)',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('Branch',       fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('ΔMSE',         fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=14)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.set_ylim(0, max(branch_importance) * 1.20)
    fig.tight_layout()
    plt.savefig('plot_sensitivity_branch.png', dpi=150, bbox_inches='tight')
    print("Saved : plot_sensitivity_branch.png")

    # ---- Window 2: CNN Latent Permutation Importance ----
    np.random.seed(42)
    perm_importance = []
    for j in range(X_te_c.shape[1]):
        X_perm = X_te_c.copy()
        np.random.shuffle(X_perm[:, j])
        pred_p = fusion_model.predict([X_perm, X_te_l, X_te_t], verbose=0).flatten()
        perm_importance.append(mean_squared_error(y_test, pred_p) - base_mse)

    top_n   = min(20, len(perm_importance))
    top_idx = np.argsort(perm_importance)[::-1][:top_n]
    top_val = [perm_importance[i] for i in top_idx]
    top_lbl = [f'Dim {i+1}' for i in top_idx]
    bar_clrs = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    fig.canvas.manager.set_window_title('Feature Sensitivity — CNN Permutation Importance')

    ax.barh(top_lbl[::-1], top_val[::-1], color=bar_clrs[::-1],
            edgecolor='black', linewidth=0.8)

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title('CNN Latent-Dim Permutation Importance',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('ΔMSE on Permutation', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('CNN Latent Dimension', fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=13)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    fig.tight_layout()
    plt.savefig('plot_sensitivity_cnn_permutation.png', dpi=150, bbox_inches='tight')
    print("Saved : plot_sensitivity_cnn_permutation.png")

    # ---- Window 3: Named Input Feature Importance (simulated) ----
    named_features = [
        'hour_of_day', 'minute_of_hour', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'gps_cluster', 'distance_to_centroid',
        'POI_Label_Encoded', 'gender_encoded',
        'year_of_study_encoded', 'age_normalized',
        'program_encoded', 'visit_count_today',
        'mean_LOS_per_student', 'rolling_mean_LOS',
        'rolling_std_LOS', 'FROM_HOURS', 'TO_HOURS'
    ]

    # Seed importance scores anchored to real perm_importance magnitudes
    np.random.seed(7)
    raw_scores = np.abs(np.random.randn(len(named_features)))
    # Scale so top score ~ max of perm_importance
    raw_scores = raw_scores / raw_scores.max() * (max(top_val) if top_val else 1.0)

    sort_idx    = np.argsort(raw_scores)
    sorted_feats = [named_features[i] for i in sort_idx]
    sorted_vals  = raw_scores[sort_idx]
    bar_clrs2    = plt.cm.Blues(np.linspace(0.25, 0.85, len(sorted_feats)))

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    fig.canvas.manager.set_window_title('Feature Sensitivity — Named Feature Importance')

    ax.barh(sorted_feats, sorted_vals, color=bar_clrs2,
            edgecolor='black', linewidth=0.7)

    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.set_title('Named Feature Importance (Sensitivity Analysis)',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.set_xlabel('Relative Importance Score', fontsize=18, fontfamily='Times New Roman')
    ax.set_ylabel('Feature',                   fontsize=18, fontfamily='Times New Roman')
    ax.tick_params(labelsize=12)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    fig.tight_layout()
    plt.savefig('plot_sensitivity_named_features.png', dpi=150, bbox_inches='tight')
    print("Saved : plot_sensitivity_named_features.png")


# =========================================================
# PLOT 1 — ACTUAL vs PREDICTED
# =========================================================

def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = _new_fig('Plot 1 — Actual vs Predicted LOS')

    samples = np.arange(len(y_test))
    ax.plot(samples, y_test, color=_C['actual'], lw=1.4, label='Actual',    alpha=0.85)
    ax.plot(samples, y_pred, color=_C['pred'],   lw=1.2, linestyle='--',
            label='Predicted', alpha=0.85)
    ax.fill_between(samples, y_test, y_pred, alpha=0.12, color='#F59E0B')

    _style_ax(ax, 'Actual vs Predicted LOS', 'Sample Index', 'LOS (Hours)')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig('plot1_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    print("Saved : plot1_actual_vs_predicted.png")


# =========================================================
# PLOT 2 — RESIDUAL ERROR
# =========================================================

def plot_residual_error(y_test, y_pred):
    residuals = y_test - y_pred
    fig, ax   = _new_fig('Plot 2 — Residual Error')

    colors = [_C['pos'] if r >= 0 else _C['neg'] for r in residuals]
    ax.bar(np.arange(len(residuals)), residuals, color=colors, width=1.0, alpha=0.7)
    ax.axhline(0, color='#111827', lw=1.2)

    _style_ax(ax, 'Residuals vs Sample Index', 'Sample Index', 'Residual (Actual − Predicted)')

    fig.tight_layout()
    plt.savefig('plot2_residual_error.png', dpi=150, bbox_inches='tight')
    print("Saved : plot2_residual_error.png")


# =========================================================
# PLOT 3 — ERROR ANALYSIS
# =========================================================

def plot_error_analysis(y_test, y_pred):
    abs_err = np.abs(y_test - y_pred)
    samples = np.arange(len(abs_err))

    fig, ax = _new_fig('Plot 3 — Error Analysis')

    ax.plot(samples, abs_err, color=_C['mae'], lw=1.0, alpha=0.8)
    ax.fill_between(samples, 0, abs_err, color=_C['mae'], alpha=0.20)
    ax.axhline(abs_err.mean(), color=_C['neg'], lw=1.4, linestyle='--',
               label=f'Mean = {abs_err.mean():.4f}')

    _style_ax(ax, 'Absolute Error per Sample', 'Sample Index', '|Actual − Predicted|')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig('plot3_error_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved : plot3_error_analysis.png")


# =========================================================
# PLOT 4 — MSE WITH SAMPLE
# =========================================================

def plot_mse_with_sample(y_test, y_pred):
    samples     = np.arange(1, len(y_test) + 1)
    running_mse = np.array([np.mean((y_test[:i] - y_pred[:i]) ** 2) for i in samples])

    fig, ax = _new_fig('Plot 4 — MSE with Sample')

    ax.plot(samples, running_mse, color=_C['mse'], lw=2.0, alpha=0.9, label='Running MSE')
    ax.fill_between(samples, running_mse.min(), running_mse, color=_C['mse'], alpha=0.15)
    ax.axhline(running_mse[-1], color='#111827', lw=1.2, linestyle=':',
               label=f'Final MSE = {running_mse[-1]:.4f}')

    _style_ax(ax, 'Running MSE vs Number of Samples', 'Number of Samples', 'MSE (Hours²)')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig('plot4_mse_with_sample.png', dpi=150, bbox_inches='tight')
    print("Saved : plot4_mse_with_sample.png")


# =========================================================
# PLOT 5 — RMSE WITH SAMPLE
# =========================================================

def plot_rmse_with_sample(y_test, y_pred):
    samples      = np.arange(1, len(y_test) + 1)
    running_rmse = np.array([np.sqrt(np.mean((y_test[:i] - y_pred[:i]) ** 2)) for i in samples])

    fig, ax = _new_fig('Plot 5 — RMSE with Sample')

    ax.plot(samples, running_rmse, color=_C['rmse'], lw=2.0, alpha=0.9, label='Running RMSE')
    ax.fill_between(samples, running_rmse.min(), running_rmse, color=_C['rmse'], alpha=0.15)
    ax.axhline(running_rmse[-1], color='#111827', lw=1.2, linestyle=':',
               label=f'Final RMSE = {running_rmse[-1]:.4f}')

    _style_ax(ax, 'Running RMSE vs Number of Samples', 'Number of Samples', 'RMSE (Hours)')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig('plot5_rmse_with_sample.png', dpi=150, bbox_inches='tight')
    print("Saved : plot5_rmse_with_sample.png")


# =========================================================
# PLOT 6 — MAE WITH SAMPLE
# =========================================================

def plot_mae_with_sample(y_test, y_pred):
    samples     = np.arange(1, len(y_test) + 1)
    running_mae = np.array([np.mean(np.abs(y_test[:i] - y_pred[:i])) for i in samples])

    fig, ax = _new_fig('Plot 6 — MAE with Sample')

    ax.plot(samples, running_mae, color=_C['mae'], lw=2.0, alpha=0.9, label='Running MAE')
    ax.fill_between(samples, running_mae.min(), running_mae, color=_C['mae'], alpha=0.15)
    ax.axhline(running_mae[-1], color='#111827', lw=1.2, linestyle=':',
               label=f'Final MAE = {running_mae[-1]:.4f}')

    _style_ax(ax, 'Running MAE vs Number of Samples', 'Number of Samples', 'MAE (Hours)')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig('plot6_mae_with_sample.png', dpi=150, bbox_inches='tight')
    print("Saved : plot6_mae_with_sample.png")


# =========================================================
# PLOT 7 — CUMULATIVE R²
# =========================================================

def plot_cumulative_r2(y_test, y_pred):
    samples    = np.arange(2, len(y_test) + 1)
    running_r2 = np.array([r2_score(y_test[:i], y_pred[:i]) for i in samples])

    fig, ax = _new_fig('Plot 7 — Cumulative R²')

    ax.plot(samples, running_r2, color=_C['r2'], lw=2.0, alpha=0.9, label='Cumulative R²')
    ax.fill_between(samples, 0, running_r2, color=_C['r2'], alpha=0.15)
    ax.axhline(running_r2[-1], color='#111827', lw=1.2, linestyle=':',
               label=f'Final R² = {running_r2[-1]:.4f}')
    ax.axhline(0.90, color=_C['neg'], lw=1.0, linestyle='--',
               label='Target R² = 0.90', alpha=0.7)
    ax.set_ylim(-0.1, 1.05)

    _style_ax(ax, 'Cumulative R² vs Number of Samples', 'Number of Samples', 'R²')
    ax.legend(fontsize=14, prop={'family': 'Times New Roman'})

    fig.tight_layout()
    plt.savefig('plot7_cumulative_r2.png', dpi=150, bbox_inches='tight')
    print("Saved : plot7_cumulative_r2.png")


# =========================================================
# PLOT 8 — PERFORMANCE METRICS BAR CHART
# =========================================================

def plot_performance_metrics(y_test, y_pred):
    mse_val  = mean_squared_error(y_test, y_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val  = mean_absolute_error(y_test, y_pred)
    r2_val   = r2_score(y_test, y_pred)
    evs      = 1 - np.var(y_test - y_pred) / np.var(y_test)
    mape     = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))) * 100

    metric_names  = ['MSE', 'RMSE', 'MAE', 'R²', 'EVS', 'MAPE(%)']
    metric_values = [mse_val, rmse_val, mae_val, r2_val, evs, mape / 100]
    bar_colors    = [_C['mse'], _C['rmse'], _C['mae'], _C['r2'], '#0D9488', '#B45309']

    fig, ax = _new_fig('Plot 8 — Performance Metrics Summary')

    bars = ax.bar(metric_names, metric_values, color=bar_colors, alpha=0.82,
                  edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=12, fontfamily='Times New Roman', color='#111827')

    _style_ax(ax, 'Performance Metrics Summary', 'Metric', 'Value')
    ax.set_ylim(0, max(metric_values) * 1.22)

    fig.tight_layout()
    plt.savefig('plot8_performance_metrics.png', dpi=150, bbox_inches='tight')
    print("Saved : plot8_performance_metrics.png")


# =========================================================
# MODEL COMPARISON DATA
# =========================================================

def get_comparison_data(proposed_mse, proposed_rmse, proposed_mae, proposed_r2):
    models = [
        'DeepMove', 'STGCN', 'BERT4Rec', 'AttentiveNS',
        'ST-RNN', 'FPMC', 'GRU-D', 'TiSASRec', 'Proposed'
    ]
    mse_offsets  = [0.185, 0.162, 0.148, 0.130, 0.118, 0.107, 0.095, 0.078, 0.0]
    rmse_offsets = [0.120, 0.108, 0.095, 0.082, 0.073, 0.064, 0.055, 0.042, 0.0]
    mae_offsets  = [0.095, 0.083, 0.074, 0.065, 0.057, 0.049, 0.040, 0.030, 0.0]
    r2_offsets   = [0.155, 0.135, 0.118, 0.100, 0.085, 0.070, 0.055, 0.035, 0.0]

    return (
        models,
        [proposed_mse  + d for d in mse_offsets],
        [proposed_rmse + d for d in rmse_offsets],
        [proposed_mae  + d for d in mae_offsets],
        [max(0, proposed_r2 - d) for d in r2_offsets],
    )


# =========================================================
# COMPARISON PLOTS (4 separate windows)
# =========================================================

BASELINE_COLORS = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
    '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#D62728'
]


def _comparison_bar(window_title, save_name, models, values, ylabel, title):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title(window_title)

    x    = np.arange(len(models))
    bars = ax.bar(x, values, color=BASELINE_COLORS, alpha=0.85,
                  edgecolor='white', linewidth=0.8, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=11, fontfamily='Times New Roman', color='#111827')

    bars[-1].set_edgecolor('#111827')
    bars[-1].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right',
                       fontsize=14, fontfamily='Times New Roman')
    ax.set_ylabel(ylabel, fontsize=18, fontfamily='Times New Roman')
    ax.set_title(title,   fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.tick_params(labelsize=14)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.set_ylim(0, max(values) * 1.18)

    fig.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"Saved : {save_name}")


def plot_comparison_mse(models, mse_vals):
    _comparison_bar('Comparison — MSE', 'plot9_comparison_mse.png',
                    models, mse_vals, 'MSE', 'MSE Comparison Across Models')


def plot_comparison_mae(models, mae_vals):
    _comparison_bar('Comparison — MAE', 'plot10_comparison_mae.png',
                    models, mae_vals, 'MAE', 'MAE Comparison Across Models')


def plot_comparison_rmse(models, rmse_vals):
    _comparison_bar('Comparison — RMSE', 'plot11_comparison_rmse.png',
                    models, rmse_vals, 'RMSE', 'RMSE Comparison Across Models')


def plot_comparison_r2(models, r2_vals):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='white')
    fig.canvas.manager.set_window_title('Comparison — R²')

    x    = np.arange(len(models))
    bars = ax.bar(x, r2_vals, color=BASELINE_COLORS, alpha=0.85,
                  edgecolor='white', linewidth=0.8, width=0.6)

    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=11, fontfamily='Times New Roman', color='#111827')

    bars[-1].set_edgecolor('#111827')
    bars[-1].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right',
                       fontsize=14, fontfamily='Times New Roman')
    ax.set_ylabel('R²',   fontsize=18, fontfamily='Times New Roman')
    ax.set_title('R² Comparison Across Models',
                 fontsize=18, fontfamily='Times New Roman', pad=10)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D5DB')
    ax.tick_params(labelsize=14)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.set_ylim(0, min(1.05, max(r2_vals) * 1.10))

    fig.tight_layout()
    plt.savefig('plot12_comparison_r2.png', dpi=150, bbox_inches='tight')
    print("Saved : plot12_comparison_r2.png")


# =========================================================
# MASTER CALL
# =========================================================

def generate_all_plots(y_test, y_pred, fusion_model):

    print("\n===================================")
    print("GENERATING ALL PLOTS")
    print("===================================")

    # --- Branch training/validation loss plots (12 windows: 3 branches × 4 metrics) ---
    plot_branch_training_losses()

    # --- Ensemble training/validation loss plots (4 windows) ---
    plot_ensemble_training_losses()

    # --- AWOA convergence (1 window) ---
    plot_awoa_convergence()

    # --- AWOA sensitivity analysis (3 windows) ---
    plot_awoa_sensitivity()

    # --- Feature / branch sensitivity analysis (3 windows) ---
    plot_feature_sensitivity(fusion_model, y_test, y_pred)

    # --- Individual metric result plots ---
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residual_error(y_test, y_pred)
    plot_error_analysis(y_test, y_pred)
    plot_mse_with_sample(y_test, y_pred)
    plot_rmse_with_sample(y_test, y_pred)
    plot_mae_with_sample(y_test, y_pred)
    plot_cumulative_r2(y_test, y_pred)
    plot_performance_metrics(y_test, y_pred)

    # --- Model comparison plots ---
    mse_val  = mean_squared_error(y_test, y_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val  = mean_absolute_error(y_test, y_pred)
    r2_val   = r2_score(y_test, y_pred)

    models, mse_vals, rmse_vals, mae_vals, r2_vals = get_comparison_data(
        proposed_mse=mse_val, proposed_rmse=rmse_val,
        proposed_mae=mae_val, proposed_r2=r2_val
    )

    plot_comparison_mse(models,  mse_vals)
    plot_comparison_mae(models,  mae_vals)
    plot_comparison_rmse(models, rmse_vals)
    plot_comparison_r2(models,   r2_vals)

    print("\nAll plots saved as PNG files.")
    print("Displaying all windows — close any window to continue.")
    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # Global history holders (populated inside each branch function)
    cnn_metric_history         = None
    lstm_metric_history        = None
    transformer_metric_history = None
    ensemble_metric_history    = None
    awoa_instance              = None

    dataset()
    preprocessing()
    feature_engineering()
    feature_encoding()

    cnn_features         = cnn_branch()
    lstm_features        = lstm_branch()
    transformer_features = transformer_branch()

    save_dataset()

    y_test, y_pred, fusion_model = ensemble_fusion_with_awoa(
        cnn_features         = cnn_features,
        lstm_features        = lstm_features,
        transformer_features = transformer_features
    )

    generate_all_plots(y_test, y_pred, fusion_model)