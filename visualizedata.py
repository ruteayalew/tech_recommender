import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import numpy as np
import os


# Wrapper function that calls all the data visualization functions
def print_visuals(df, user_id, item_id, target_attribute, random_state, num_users, num_items):
    ratings_distribution(df, user_id, item_id)
    correlation_heatmap(df, user_id, item_id, target_attribute, random_state, num_users, num_items)

# Function that calls the two functions relating to the distribution of ratings
def ratings_distribution(df, user_id, item_id):
    # visualize ratings per user
    ratings_per_user(df, user_id)

    # visualize ratings per item
    ratings_per_item(df, item_id)

# Function to display the number of ratings per user
def ratings_per_user(df, user_id):
    # Get the number of unique users
    unique_ids = df[user_id].nunique()
    print("Number of unique users: ", unique_ids)

    # Group users by the exact number of ratings
    grouped_users = df.groupby(user_id).size()

    # Plotting users grouped by the exact number of ratings
    plt.figure(figsize=(10, 4))
    grouped_users.value_counts().sort_index().plot(kind='bar', width=0.8, color='c')
    plt.title('Number of Users per Group Based on Exact Number of Ratings')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.show()

    # Get the average of the ratings each user has given
    average_ratings = df.groupby(user_id)['rating'].mean()

    # Plotting the number of users in each bin
    plt.figure(figsize=(10, 4))
    bins = pd.cut(average_ratings, bins=5)  # Adjust the number of bins as needed
    user_counts_per_bin = bins.value_counts().sort_index()
    user_counts_per_bin.plot(kind='bar', width=0.8, color='m')
    plt.title('Number of Users per User Group Based on Average Ratings')
    plt.xlabel('User Groups (Based on Average Ratings)')
    plt.ylabel('Number of Users')
    plt.show()

    print('Number of users per user group:\n', user_counts_per_bin)

# Function to display the number of ratings per item
def ratings_per_item(df, item_id):
    # Get the number of unique users
    unique_ids = df[item_id].nunique()
    print("Number of unique items: ", unique_ids)

    # Group items by the exact number of ratings
    grouped_users = df.groupby(item_id).size()

    # Plotting users grouped by the exact number of ratings
    plt.figure(figsize=(10, 4))
    ax = grouped_users.value_counts().sort_index().plot(kind='bar', width=0.8, color='c')
    plt.title('Number of Items per Group Based on Exact Number of Ratings')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Items')

    # If number of ratings is large, set x-axis ticks at intervals of 5
    num_ticks = len(ax.get_xticks())
    if num_ticks>50:
        x_ticks = [tick for tick in ax.get_xticks() if tick % 5 == 0]
        ax.set_xticks(x_ticks)
    
    plt.show()
    

    # Get the average of the ratings each user has given
    average_ratings = df.groupby(item_id)['rating'].mean()

    # Plotting the number of items in each bin
    plt.figure(figsize=(10, 4))
    bins = pd.cut(average_ratings, bins=5) 
    user_counts_per_bin = bins.value_counts().sort_index()
    user_counts_per_bin.plot(kind='bar', width=0.8, color='m')
    plt.title('Number of Items per Item Group Based on Average Ratings')
    plt.xlabel('Item Groups (Based on Average Ratings)')
    plt.ylabel('Number of Items')
    plt.show()

    print('Number of items per item group:\n', user_counts_per_bin)

# Function to create a heatmap of correlation
def correlation_heatmap(df, user_id, item_id, target_attribute, random_state, num_users, num_items):
    random.seed(random_state)
    # Sample a subset of users and items
    sampled_users = random.sample(df[user_id].unique().tolist(), k=num_users)
    sampled_items = random.sample(df[item_id].unique().tolist(), k=num_items)

    # Filter the data to include only the sampled users and items
    filtered_data = df[(df[user_id].isin(sampled_users)) & (df[item_id].isin(sampled_items))]

    # Pivot the data to create a user-item interaction matrix
    user_item_matrix = filtered_data.pivot_table(index=user_id, columns=item_id, values=target_attribute, fill_value=0)

    # Calculate the correlation matrix between items
    item_correlation = user_item_matrix.corr()

    # Create a heatmap of the item correlation matrix
    plt.figure(figsize=(15, 10))
    sns.heatmap(item_correlation, cmap='jet', annot=False, fmt=".2f", cbar_kws={'label': 'Correlation'})
    plt.title('Item Correlation Heatmap (Sampled)')
    plt.show()

