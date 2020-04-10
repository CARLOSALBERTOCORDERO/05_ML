import pandas as pd
import numpy as np

# Create dataframe
data = pd.DataFrame()
data['Gender'] = ['male','male','male','male','female',
        'female','female','female']
data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_Size'] = [12,11,12,10,6,8,7,9]
print("\nData frame:\n", data)

# Create input dataframe
person = pd.DataFrame()
person['Height'] = [6]
person['Weight'] = [130]
person['Foot_Size'] = [8]
print("\nPerson dataframe:\n", person)

# Calculate Priors
n_male = data['Gender'][data['Gender'] == 'male'].count()
n_female = data['Gender'][data['Gender'] == 'female'].count()
total_ppl = n_male + n_female       # data['Gender'].count()
P_male = n_male/total_ppl
P_female = n_female/total_ppl

# Calcualte mean and variance of each feature per class
data_means = data.groupby('Gender').mean()
print("\nMeans by class:\nn", data_means)
data_variance = data.groupby('Gender').var()
print("\nVariance by class:\n", data_variance)

# Means for male and female
male_height_mean = (data_means['Height']
        [data_variance.index == 'male'].values[0])
male_weight_mean = (data_means['Weight']
        [data_variance.index == 'male'].values[0])
male_footsize_mean = (data_means['Foot_Size']
        [data_variance.index == 'male'].values[0])
female_height_mean = (data_means['Height']
        [data_variance.index == 'female'].values[0])
female_weight_mean = (data_means['Weight']
        [data_variance.index == 'female'].values[0])
female_footsize_mean = (data_means['Foot_Size']
        [data_variance.index == 'female'].values[0])

# Variances for male and female
male_height_variance = (data_variance['Height']
        [data_variance.index == 'male'].values[0])
male_weight_variance = (data_variance['Weight']
        [data_variance.index == 'male'].values[0])
male_footsize_variance = (data_variance['Foot_Size']
        [data_variance.index == 'male'].values[0])
female_height_variance = (data_variance['Height']
        [data_variance.index == 'female'].values[0])
female_weight_variance = (data_variance['Weight']
        [data_variance.index == 'female'].values[0])
female_footsize_variance = (data_variance['Foot_Size']
        [data_variance.index == 'female'].values[0])


# Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):
    return (1.0 / (np.sqrt(2 * np.pi * variance_y)) * 
            np.exp((-(x - mean_y) ** 2) / (2 * variance_y)))


numerator_for_male = (P_male * 
    p_x_given_y(person['Height'][0], 
    male_height_mean, male_height_variance) *
    p_x_given_y(person['Weight'][0], 
    male_weight_mean, male_weight_variance) *
    p_x_given_y(person['Foot_Size'][0], 
    male_footsize_mean, male_footsize_variance))
print("\n  MALE:", format(numerator_for_male, '12.4e'))

numerator_for_female = (P_female *
        p_x_given_y(person['Height'][0], 
                female_height_mean, female_height_variance) * 
        p_x_given_y(person['Weight'][0], 
                female_weight_mean, female_weight_variance) *
        p_x_given_y(person['Foot_Size'][0], 
                female_footsize_mean, female_footsize_variance))
print("\nFEMALE:", format(numerator_for_female, '12.4e'))

