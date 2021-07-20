# Converting TensorFlow* Wide and Deep Family Models to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_WideAndDeep_Family_Models}

The Wide and Deep models is a combination of wide and deep parts for memorization and generalization of object features respectively.
These models can contain different types of object features such as numerical, categorical, sparse and sequential features. These feature types are specified
through Tensorflow* tf.feature_column API. Table below presents what feature types are supported by the OpenVINO&trade; toolkit.

| numeric | (weighted) categorical | categorical with hash | bucketized | sequential | crossed |
|:-------:|:----------------------:|:---------------------:|:----------:|:----------:|:-------:|
| yes     | yes                    | no                    | yes        | yes        | no      |

**NOTE**: the categorical with hash and crossed features are currently unsupported since The OpenVINO&trade; toolkit does not support tensors of `string` type and operations with them.

## Prepare an Example of Wide and Deep Model

**Step 1**. Clone the GitHub repository with TensorFlow models and move to the directory with an example of Wide and Deep model:

```sh
git clone https://github.com/tensorflow/models.git;
cd official/r1/wide_deep
```

**Step 2**. Train the model

As the OpenVINO&trade; toolkit does not support the categorical with hash and crossed features, such feature types must be switched off in the model 
by changing the `build_model_columns()` function in `census_dataset.py` as follows:

```python
def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous variable columns
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')
  education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])
  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      'workclass', [
          'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
  # To show an example of hashing:
  #occupation = tf.feature_column.categorical_column_with_hash_bucket(
  #    'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)
  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  # Wide columns and deep columns.
  base_columns = [
      education, marital_status, relationship, workclass, 
      age_buckets,
  ]
  crossed_columns = []
  wide_columns = base_columns + crossed_columns
  deep_columns = [
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(workclass),
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      # To show an example of embedding
  ]
  return wide_columns, deep_columns
```

After that start training by the following command:

```sh
python census_main.py
```

## Convert the Wide and Deep Model to IR

Use the following command line to convert the saved model file with the checkpoint:

```sh
python mo.py 
--input_checkpoint checkpoint --input_meta_graph model.ckpt.meta
--input "IteratorGetNext:0[2],
         IteratorGetNext:1[2],
         IteratorGetNext:2[2],
         IteratorGetNext:4[2],
         IteratorGetNext:7[2],
         linear/linear_model/linear_model/linear_model/education/to_sparse_input/indices:0[10 2]{i32},
         linear/linear_model/linear_model/linear_model/education/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         linear/linear_model/linear_model/linear_model/education/to_sparse_input/dense_shape:0[2]{i32}->[2 50],
         linear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/indices:0[10 2]{i32},
         linear/linear_model/linear_model/linear_model/marital_status/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         linear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/dense_shape:0[2]{i32}->[2 50],
         linear/l inear_model/linear_model/linear_model/relationship/to_sparse_input/indices:0[10 2]{i32},
         linear/linear_model/linear_model/linear_model/relationship/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         linear/linear_model/linear_model/linear_model/relationship/to_sparse_input/dense_shape:0[2]{i32}->[2 50],
         linear/linear_model/linear_model/linear_model/workclass/to_sparse_input/indices:0[10 2]{i32},
         linear/linear_model/linear_model/linear_model/workclass/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         linear/linear_model/linear_model/linear_model/workclass/to_sparse_input/dense_shape:0[2]{i32}->[2 50],
         dnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/indices:0[10 2]{i32},
         dnn/input_from_feature_columns/input_layer/education_indicator/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         dnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/dense_shape:0[2]{i32}->[2 50],
         dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/indices:0[10 2]{i32},
         dnn/input_from_feature_columns/input_layer/marital_status_indicator/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/dense_shape:0[2]{i32}->[2 50],
         dnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/indices:0[10 2]{i32},
         dnn/input_from_feature_columns/input_layer/relationship_indicator/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         dnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/dense_shape:0[2]{i32}->[2 50],
         dnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/indices:0[10 2]{i32},
         dnn/input_from_feature_columns/input_layer/workclass_indicator/hash_table_Lookup/LookupTableFindV2:0[10]{i32},
         dnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/dense_shape:0[2]{i32}->[2 50]" 
--output head/predictions/probabilities
```

The model contains operations unsupported by the OpenVINO&trade; toolkit such as `IteratorGetNext` and `LookupTableFindV2`, so the Model Optimizer must prune these nodes.
The pruning is specified through `--input` option. The prunings for `IteratorGetNext:*` nodes correspond to numeric features.
The pruning for each categorical feature consists of three prunings for the following nodes: `*/to_sparse_input/indices:0`, `*/hash_table_Lookup/LookupTableFindV2:0`, and `*/to_sparse_input/dense_shape:0`.

The above command line generates IR for a batch of two objects, with total number of actual categorical feature values equal to 10 and maximum size of sparse categorical feature for one object equal to 50.
