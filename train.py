streamlit
numpy
pandas
scikit-learn
pickle-mixin
tensorflow
matplotlib 
seaborn

# Load dataset
df = pd.read_csv("lab_11_bridge_data.csv")

# Data preprocessing
categorical_features = ["Material"]
numerical_features = ["Span_ft", "Deck_Width_ft", "Age_Years", "Num_Lanes", "Condition_Rating"]

ohe = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_features),
        ('cat', ohe, categorical_features)
    ]
)

X = df.drop(columns=["Bridge_ID", "Max_Load_Tons"])
y = df["Max_Load_Tons"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Save the preprocessor
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

# ANN Model
def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train.shape[1])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=100, 
    batch_size=16,
    callbacks=[early_stopping]
)

# Save the model
model.save("tf_bridge_model.h5")
