from data import *
import scipy as sp
from ucimlrepo import fetch_ucirepo

# fetch dataset to create pre-processed CSV files


def abaloneCreateData():
    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features
    X = X.drop("Sex", axis=1)
    X.to_csv("datasets/abalone.csv")
    X.to_csv("datasets/abaloneNoIndex.csv", index=None, header=None)


def electricityCreateData():
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)
    X = individual_household_electric_power_consumption.data.features
    firstMonth = (
        X[X.Date.str.contains("/12/2006")].drop("Date", axis=1).drop("Time", axis=1)
    )
    X.drop("Date", axis=1).drop("Time", axis=1)
    X.to_csv("datasets/electricity,csv")
    X.to_csv("datasets/electricityNoIndex.csv", index=None, header=None)
    firstMonth.to_csv("datasets/electricityFirstMonth.csv", index=None, header=None)


def onlineRetailCreateData():
    online_retail = fetch_ucirepo(id=352)
    X = online_retail.data.features
    X.to_csv("datasets/onlineRetail.csv")
    firstDay = X[X.InvoiceDate.str.contains("12/(?:1|2)/2010")][
        ["Quantity", "UnitPrice"]
    ]
    firstDay.to_csv("datasets/onlineRetailFirstDay.csv")  ##
    X = X[["Quantity", "UnitPrice"]]
    X.to_csv("datasets/onlineRetailNoIndex.csv", index=None, header=None)


def angularToCartesian(lat, long):
    R = 6371
    x = np.cos(lat) * np.cos(long) * R
    y = np.cos(lat) * np.sin(long) * R
    z = np.sin(lat)
    return x, y, z


def uberTwoDaysPreprocess(maxRows=-1):
    u = pd.read_csv("datasets/uber/uber-raw-data-jun14.csv")
    firstDay = (
        u[u["Date/Time"].str.contains("6/1/2014")]
        .drop("Date/Time", axis=1)
        .drop("Base", axis=1)
    )
    secondDay = (
        u[u["Date/Time"].str.contains("6/2/2014")]
        .drop("Date/Time", axis=1)
        .drop("Base", axis=1)
    )

    n, m = len(firstDay["Lat"]), len(secondDay["Lat"])
    firstDay = firstDay.iloc[:n, :].set_index(np.arange(n))
    secondDay = secondDay.iloc[:m, :].set_index(np.arange(m))
    if n > m:
        n, m = m, n
        firstDay, secondDay = secondDay, firstDay
    if maxRows != -1:
        n, m = maxRows, maxRows
    arr = np.zeros((n, m))
    print("Computing adjacency for matching, this might take a while.")
    for i in range(n):
        if i % 100 == 0:
            print(i / n * 100, "%")
        for j in range(m):
            lat1, long1 = firstDay["Lat"][i], firstDay["Lon"][i]
            x1, y1, z1 = angularToCartesian(lat1, long1)
            lat2, long2 = secondDay["Lat"][j], secondDay["Lon"][j]
            x2, y2, z2 = angularToCartesian(lat2, long2)
            arr[i, j] = dist(np.array([[x1, y1, z1], [x2, y2, z2]]), 0, 1)
    print("Done.")
    matching = match(
        n, m, arr
    )  # matching from first day to second day. matching[i] is matched place of i in the second day
    pointsDayOne = []
    pointsDayTwo = []
    nbPointsMached = 0
    for i in matching.keys():
        if arr[i, matching[i]] > 1:
            continue
        nbPointsMached += 1
        pointsDayOne.append(i)
        pointsDayTwo.append(matching[i])
    firstDay = (
        firstDay.reindex(pointsDayOne)
        .iloc[:nbPointsMached, :]
        .set_index(np.arange(nbPointsMached))
    )
    secondDay = (
        secondDay.reindex(pointsDayTwo)
        .iloc[:nbPointsMached, :]
        .set_index(np.arange(nbPointsMached))
    )
    points1 = []
    points2 = []
    for i in range(nbPointsMached):
        points1.append((angularToCartesian(firstDay["Lat"][i], firstDay["Lon"][i])))
        points2.append((angularToCartesian(secondDay["Lat"][i], secondDay["Lon"][i])))
    f = open("datasets/uber/PreProcessed.csv", "w")
    f.write("x1,y1,z1,x2,y2,z2\n")
    for i in range(nbPointsMached):
        f.write(
            f"{points1[i][0]},{points1[i][1]},{points1[i][2]},{points2[i][0]},{points2[i][1]},{points2[i][2]}\n"
        )
    return np.array(firstDay), np.array(secondDay)


def match(n, m, W):
    U = np.arange(n)
    V = np.arange(m)
    left_matches = sp.optimize.linear_sum_assignment(W)
    d = {U[u]: V[v] for u, v in zip(*left_matches)}
    return d


# print("Creating csv file for Abalone dataset...")
# abaloneCreateData()
print("Creating csv file for OnlineRetail dataset...")
onlineRetailCreateData()
print("Creating csv file for Electricity dataset...")
electricityCreateData()
# print("Creating csv file for Uber dataset...")
# uberTwoDaysPreprocess()
