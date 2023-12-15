import pickle
import polars as pl
import polars.selectors as cs
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


def parse_xgb_output(output):
    results = []

    for line in output.strip().split("\n"):
        iteration_line, train_line, val_line = line.split("\t")

        iteration = int(iteration_line.strip("[]"))
        train_auc = float(train_line.split(":")[1])
        val_auc = float(val_line.split(":")[1])

        results.append((iteration, train_auc, val_auc))

    columns = ["num_iteration", "train_auc", "val_auc"]
    df_results = pl.DataFrame(results, schema=columns)

    return df_results


df = (
    pl.scan_csv("./data/bank_customer_churn.csv").select(
        pl.all().exclude("RowNumber", "CustomerId")
    )
).collect()


# Data preparation

df.columns = [column.lower() for column in df.columns]
df = df.rename(
    {
        "creditscore": "credit_score",
        "numofproducts": "num_of_products",
        "hascrcard": "has_credit_card",
        "isactivemember": "is_active_member",
        "estimatedsalary": "estimated_salary",
        "exited": "churn",
    }
)

# Setting up the validation framework

df_full_train, df_test = train_test_split(
    df, test_size=0.2, shuffle=True, random_state=11
)
y_full_train = df_full_train.select("churn").to_pandas().values
df_full_train = df_full_train.drop("churn")
y_test = df_test.select("churn").to_pandas().values
df_test = df_test.drop("churn")

# extract categorical features
categorical = df_full_train.select(cs.string()).columns
categorical.append("is_active_member")
categorical.append("has_credit_card")
categorical.remove("surname")

# extract numerical features
numerical = df_full_train.select(cs.numeric()).columns
numerical.remove("is_active_member")
numerical.remove("has_credit_card")


# Model training

dv = DictVectorizer(sparse=False)

# create feature matrix
dicts_full_train = df_full_train.select(categorical + numerical).to_dicts()
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.select(categorical + numerical).to_dicts()
X_test = dv.fit_transform(dicts_test)

features = list(dv.get_feature_names_out())

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

xgb_params = {
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 10,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}
model = xgb.train(xgb_params, dfulltrain, num_boost_round=155)


# Save model

output_file = "xgboost_model.bin"
with open(output_file, "wb") as file_out:
    pickle.dump((dv, model), file_out)
