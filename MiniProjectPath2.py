import pandas 
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge





def highest_traffic(data):

    brooklyn_traffic = data['Brooklyn Bridge']
    manhattan_traffic = data['Manhattan Bridge']
    williamsburg_traffic = data['Williamsburg Bridge']
    queensboro_traffic = data['Queensboro Bridge']

    brooklyn_total = brooklyn_traffic.sum()
    manhattan_total = manhattan_traffic.sum()
    williamsburg_total = williamsburg_traffic.sum()
    queensboro_total = queensboro_traffic.sum()


    bridge_totals = {
        'Brooklyn Bridge': brooklyn_total,
        'Manhattan Bridge': manhattan_total,
        'Williamsburg Bridge': williamsburg_total,
        'Queensboro Bridge': queensboro_total
    }
    sorted_bridges = sorted(bridge_totals, key=bridge_totals.get, reverse=True)
    selected_bridges = sorted_bridges[:3]
    return selected_bridges

def pred_bikers(data):
    
    scores = []
    for _ in range(2000):
        X = data[['High Temp', 'Low Temp', 'Precipitation']]
        y = data['Total']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=numpy.random.randint(1000))
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
    avg_score = numpy.mean(scores)
    return avg_score

if __name__ == "__main__":
    dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
    dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
    dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
    dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
    dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))

    selected_bridges = highest_traffic(dataset_2)
    avgscore = pred_bikers(dataset_2)
    print("Average R2 score: ", avgscore)

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
