# Sentence-Classification-CNN

### Цель проекта

Целью данного проекта является классифицировать вопросы по 6 категориям. 

### Датасет

Датасет называется Experimental Data for Question Classification (https://cogcomp.seas.upenn.edu/Data/QA/QC/). Содержит ~5500 вопросов для обучения, 500 для теста.

Пример из датасета: HUM:title What is the oldest profession ?

Здесь HUM - категория, title - подкатегория, What is the oldest profession ? - сам вопрос.

### Токенизация текста

Для токенизации текста воспользуемся токенизатором из tensorflow. 

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['question'].tolist())
n_vocab = len(tokenizer.index_word) + 1
print(f"Vocabluary size: {n_vocab}")
```

```
Vocabluary size: 7917
```

Размер словаря составил 7917 токенов. 

Далее преобразуем текст в последовательность токенов.

```python
train_sequences = tokenizer.texts_to_sequences(train_df['question'].tolist())
train_labels = train_df['category'].values

valid_sequences = tokenizer.texts_to_sequences(valid_df['question'].tolist())
valid_labels = valid_df['category'].values

test_sequences = tokenizer.texts_to_sequences(test_df['question'].tolist())
test_labels = test_df['category'].values
```

### Выравнивание текста

Для того, чтобы каждый вопрос был одинаковой длины, мы применим функцию tf.keras.preprocessing.sequence.pad_sequences.

```python
from functools import partial

max_seq_length = 22

preprocessed_res = partial(
    tf.keras.preprocessing.sequence.pad_sequences,
    maxlen=max_seq_length, padding='post', truncating='post')

preprocessed_train_sequences = preprocessed_res(train_sequences)
preprocessed_valid_sequences = preprocessed_res(valid_sequences)
preprocessed_test_sequences = preprocessed_res(test_sequences)
```

Здесь мы ограничили длину вопросами 22 токенами, будем обрезать или добавлять лишние токены с конца. Сделали частичную функцию и применили её к тренировочному, валидационному и тестовому датасету.

### Модель

Для начала мы определим входной слой и слой векторизации.

```python
# Input layer takes word IDs as inputs
word_id_inputs = layers.Input(shape=(max_seq_length,), dtype='int32')
# Get the embeddings of the inputs / out [batch_size, sent_length,
# output_dim]
embedding_out = layers.Embedding(input_dim=n_vocab, output_dim=64)(word_id_inputs)
```

Размер каждого вектора будет равен 64.

Далее мы определим 3 слоя конволюции, причём слои друг от друга будут получать входные данные независимо.

```python
conv1_1 = layers.Conv1D(100, kernel_size=3, 
                        strides=1, padding='same', 
                        activation='relu')(embedding_out)
conv1_2 = layers.Conv1D(100, kernel_size=4, 
                        strides=1, padding='same', 
                        activation='relu')(embedding_out)
conv1_3 = layers.Conv1D(100, kernel_size=5, 
                        strides=1, padding='same', 
                        activation='relu')(embedding_out)
```

Мы так сделали по следующей причине: Это ведёт к **улавливанию различных n-граммных признаков**: Различные конволюционные слои с разным размером ядра предназначены для захвата различных n-граммных признаков из входного текста.

- conv1_1 с размером ядра 3 будет захватывать признаки триграммы (последовательности из трех последовательных слов).
- conv1_2 с размером ядра 4 будет фиксировать 4-граммовые признаки.
- conv1_3 с размером ядра 5 будет фиксировать 5-граммовые признаки.

Используя различные размеры ядра, сеть может научиться распознавать паттерны на разных уровнях входного текста. Например, она может распознавать как короткие фразы, так и более длинную контекстную информацию.

Это приводит к **повышению экспрессивности модели**: Каждый конволюционный слой может изучать различные паттерны и особенности входного текста. Это повышает выразительность модели и ее способность извлекать из текста значимую информацию.

Далее соединим 3 конволюции в один тензор, делаем max-pooling, разворачиваем тензор в одну размерность (не считая batch) и в конце определим dense слой для классификации.

```python
# in previous conv outputs / out [batch_size, sent_length, 300]
conv_out = layers.Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
pool_over_time_out = layers.MaxPool1D(pool_size=max_seq_length, 
                                      padding='valid')(conv_out)
# imply collapses all the dimensions (except the batch dimension)
# to a single dimension
flatten_out = layers.Flatten()(pool_over_time_out)

out = layers.Dense(n_classes, activation='softmax', 
                   kernel_regularizer=regularizers.l2(0.001))(flatten_out)
```

Ниже приведена диаграмма модели.

![model](/home/rugewit/Programming/MachineLearningOld/to_upload_on_github/worth_it/cnn/model.png)

### Параметры обучения

Обучение будет осуществляться следующим образом:

```python
cnn_model = Model(inputs=word_id_inputs, outputs=out)

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

cnn_model.summary()
lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=1,
    mode='auto', min_delta=0.0001, min_lr=0.000001
)

cnn_model.fit(
    preprocessed_train_sequences, train_labels,
    validation_data=(preprocessed_valid_sequences, valid_labels),
    batch_size=128,
    epochs=25,
    callbacks=[lr_reduce_callback]
)
```

Важно отметить, что мы будет использовать tf.keras.callbacks.ReduceLROnPlateau для уменьшения learning rate. Через каждые 3 эпохи будет проверяться значение val_loss и изменяться соответственно learning rate до тех пор, пока не закончится обучение или не будет достигнут min_lr.

### Результат

При запуске на тестовом датасете была получена точность 88%.
