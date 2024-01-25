# MODELING **************
# Use the best parameters to create and evaluate models
from sklearn.model_selection import train_test_split
from surprise import BaselineOnly, Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import KNNBaseline, KNNBasic, SVD, accuracy

def model_recs(trainset, testset, best_params_list, model_grid_pairs):

    for i, (model_class, _) in enumerate(model_grid_pairs):
        best_params = best_params_list[i]
        model = model_class(**best_params)

        # Train the model
        model.fit(trainset)

        # Make predictions on the test set
        predictions = model.test(testset)

        # Evaluate the model
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        print(f"\n{'='*40}\n{model_class.__name__} Evaluation\n{'='*40}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        