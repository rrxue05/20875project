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
