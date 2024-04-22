import pandas
if __name__ == "__main__":
    data = pandas.read_csv("nyc_bicycle_counts_2016.csv")
    selected_bridges = highest_traffic(data)


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

# question 2

def pred_bikers(data):
    bridge_sums = numpy.array(dataset_2['Brooklyn Bridge']) + numpy.array(dataset_2['Manhattan Bridge']) + numpy.array(dataset_2['Williamsburg Bridge']) + numpy.array(dataset_2['Queensboro Bridge'])
    bridge_averages_q2 = bridge_sums / 4
    print("Bridge Averages:", bridge_averages_q2)

    high_temp = data["High Temp"]
    low_temp = data["Low Temp"]
    precip = data["Precipitation"]

    metrics = numpy.array([high_temp, low_temp, precip])
    reg = LinearRegression(fit_intercept = True).fit(metrics, bridge_averages_q2)
    reg.coef_ 
    reg.intercept_
    y_pred = reg.predict(X_test)
    score = r2_score(bridge_averages_q2,y_pred)

    return score
''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out your data
