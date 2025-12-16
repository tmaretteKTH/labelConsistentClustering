import numpy as np
import pandas as pd
from helperFunctions import *


def getConfigTemporalEvolution(dataset, n):
    if dataset == "Uber":
        points1, points2 = uber(n)
        n = len(points1)
    return points1, points2, n


def getConfigDataArrival(dataset, n):
    if dataset == "Twitter":
        points = twitter(n)
        n = len(points)
    if dataset == "Electricity":
        points = electricity(n)
        n = len(points)
    if dataset == "OnlineRetail":
        points = onlineRetail(n)
        n = len(points)
    if dataset == "Abalone":
        points = abalone(n)
        n = len(points)
    if dataset == "Uber":
        points = uberSetup1(n)
        n = len(points)
    return points, n


### Import the pre-processed csv files


def twitter(n):
    X = pd.read_csv("datasets/twitter/twitter.csv")
    timestamp = "2013011202"
    firstTenHours = X.copy()
    firstTenHours = (
        firstTenHours[firstTenHours["timestamp"].astype(str).str.startswith(timestamp)]
        .drop("timestamp", axis=1)
        .dropna()
    )
    if n == -1:
        n = len(firstTenHours["longitude"]) - 1
    R = 6371
    firstTenHours = firstTenHours.iloc[:n, :].set_index(np.arange(n))
    firstTenHours["x"] = (
        np.cos(firstTenHours["latitude"]) * np.cos(firstTenHours["longitude"]) * R
    )
    firstTenHours["y"] = (
        np.cos(firstTenHours["latitude"]) * np.sin(firstTenHours["longitude"]) * R
    )
    firstTenHours["z"] = np.sin(firstTenHours["latitude"]) * R

    features = ["x", "y", "z"]
    feat = [pd.to_numeric(firstTenHours[f]) for f in features]
    return np.array(list(zip(*feat)))


def twitterScalability(n):
    X = pd.read_csv("datasets/twitter/twitter.csv")
    timestamp = "2013011201"  # first 1 hour
    firstTenHours = X.copy()
    firstTenHours = (
        firstTenHours[firstTenHours["timestamp"].astype(str).str.startswith(timestamp)]
        .drop("timestamp", axis=1)
        .dropna()
    )
    if n == -1:
        n = len(firstTenHours["longitude"]) - 1
    R = 6371
    firstTenHours = firstTenHours.iloc[:n, :].set_index(np.arange(n))
    firstTenHours["x"] = (
        np.cos(firstTenHours["latitude"]) * np.cos(firstTenHours["longitude"]) * R
    )
    firstTenHours["y"] = (
        np.cos(firstTenHours["latitude"]) * np.sin(firstTenHours["longitude"]) * R
    )
    firstTenHours["z"] = np.sin(firstTenHours["latitude"]) * R

    features = ["x", "y", "z"]
    feat = [pd.to_numeric(firstTenHours[f]) for f in features]
    return np.array(list(zip(*feat)))


def twitterEarly(n):
    X = pd.read_csv("datasets/twitter/twitter.csv")
    timestamp = "2013011202"  # first 1 hour
    firstTenHours = X.copy()
    firstTenHours = (
        firstTenHours[firstTenHours["timestamp"].astype(str).str.startswith(timestamp)]
        .drop("timestamp", axis=1)
        .dropna()
    )
    if n == -1:
        n = len(firstTenHours["longitude"]) - 1
    R = 6371
    firstTenHours = firstTenHours.iloc[:n, :].set_index(np.arange(n))
    firstTenHours["x"] = (
        np.cos(firstTenHours["latitude"]) * np.cos(firstTenHours["longitude"]) * R
    )
    firstTenHours["y"] = (
        np.cos(firstTenHours["latitude"]) * np.sin(firstTenHours["longitude"]) * R
    )
    firstTenHours["z"] = np.sin(firstTenHours["latitude"]) * R

    features = ["x", "y", "z"]
    feat = [pd.to_numeric(firstTenHours[f]) for f in features]
    return np.array(list(zip(*feat)))


def abalone(n):
    abal = pd.read_csv("datasets/abalone.csv")
    if n == -1:
        n = len(abal["Length"]) - 1

    abal = abal.iloc[:n, :].set_index(np.arange(n))
    features = [
        "Length",
        "Diameter",
        "Height",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
    ]
    feat = [pd.to_numeric(abal[f]) for f in features]
    return np.array(list(zip(*feat)))


def electricity(n):
    elec = pd.read_csv("datasets/electricityFirstMonth.csv")
    elec = elec[~elec.isin(["?"]).any(axis=1)]
    elec.dropna()
    if n == -1:
        n = len(elec["Global_active_power"]) - 1
    elec = elec.iloc[:n, :].set_index(np.arange(n))
    features = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    feat = [pd.to_numeric(elec[f]) for f in features]
    return np.array(list(zip(*feat)))


def electricityGetNext(n, X=pd.DataFrame(), date=[11, 2006]):
    m, y = date
    if m == 12:
        y += 1
        m = 1
    else:
        m += 1
    if X.empty:
        X = pd.read_csv("datasets/electricity.csv")
    nextMonth = X.copy()
    nextMonth = (
        nextMonth[nextMonth.Date.str.contains(f"/{m}/{y}")]
        .drop("Date", axis=1)
        .drop("Time", axis=1)
        .dropna()
    )
    nextMonth = nextMonth.iloc[:n, :].set_index(np.arange(n))
    features = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    feat = [pd.to_numeric(nextMonth[f]) for f in features]
    return X, [m, y], np.array(list(zip(*feat)))


def electricityGetN(t=50):
    m, y = [11, 2006]
    X = pd.read_csv("datasets/electricity.csv")
    maxN = len(X["Voltage"])
    for _ in range(t):
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
        nextMonth = (
            X[X.Date.str.contains(f"/{m}/{y}")]
            .drop("Date", axis=1)
            .drop("Time", axis=1)
            .dropna()
        )
        maxN = min(maxN, len(nextMonth["Voltage"]))
    return X, maxN - 1


def onlineRetail(n):
    onlineR = pd.read_csv("datasets/onlineRetailFirstDay.csv")
    if n == -1:
        n = len(onlineR["Quantity"]) - 1
    onlineR = onlineR.iloc[:n, :].set_index(np.arange(n))
    features = ["Quantity", "UnitPrice"]
    feat = [pd.to_numeric(onlineR[f]) for f in features]
    return np.array(list(zip(*feat)))


def onlineRetailGetNext(n, X=pd.DataFrame(), date=0):
    if X.empty:
        X = pd.read_csv("datasets/twitter/twitter.csv")
        X = X[["Quantity", "InvoiceDate", "UnitPrice"]]
        date = 0
    d = date
    d += 1
    m = 1
    y = 2011
    nextDay = X.copy()
    nextDay = (
        nextDay[nextDay.InvoiceDate.str.contains(f"{m}/{d}/{y}")]
        .drop("InvoiceDate", axis=1)
        .dropna()
    )
    if len(nextDay["Quantity"]) == 0:
        # skip that day
        return onlineRetailGetNext(n, X, d)
    nextDay = nextDay.iloc[:n, :].set_index(np.arange(n))
    features = ["Quantity", "UnitPrice"]
    feat = [pd.to_numeric(nextDay[f]) for f in features]
    return X, d, np.array(list(zip(*feat)))


def twitterGetNext(n, X=pd.DataFrame(), hour=0):
    if X.empty:
        X = pd.read_csv("datasets/twitter/twitter.csv")

        X = X[["longitude", "latitude", "timestamp"]]
        hour = 0
    timestamp = "20130112"
    if hour < 10:
        timestamp = f"{timestamp}0{hour}"
    else:
        timestamp = f"{timestamp}{hour}"
    nextHour = X.copy()
    nextHour = (
        nextHour[nextHour["timestamp"].astype(str).str.startswith(timestamp)]
        .drop("timestamp", axis=1)
        .dropna()
    )
    R = 6371
    nextHour = nextHour.iloc[:n, :].set_index(np.arange(n))
    nextHour["x"] = np.cos(nextHour["latitude"]) * np.cos(nextHour["longitude"]) * R
    nextHour["y"] = np.cos(nextHour["latitude"]) * np.sin(nextHour["longitude"]) * R
    nextHour["z"] = np.sin(nextHour["latitude"]) * R

    features = ["x", "y", "z"]
    feat = [pd.to_numeric(nextHour[f]) for f in features]
    return X, hour + 1, np.array(list(zip(*feat)))


def twitterGetN(t=20):
    if t > 23:
        "ERROR TOO MANY HOURS IN A DAY"
        return
    X = pd.read_csv("datasets/twitter/twitter.csv")

    X = X[["longitude", "latitude", "timestamp"]]
    hour = 0
    timestampDate = "20130112"
    maxN = len(X["longitude"])
    for _ in range(t):
        if hour < 10:
            timestamp = f"{timestampDate}0{hour}"
        else:
            timestamp = f"{timestampDate}{hour}"
        nextHour = X.copy()
        nextHour = (
            nextHour[nextHour["timestamp"].astype(str).str.startswith(timestamp)]
            .drop("timestamp", axis=1)
            .dropna()
        )
        maxN = min(maxN, len(nextHour["longitude"]))
        hour += 1
    return X, maxN - 1


def onlineRetailGetN(t=20):
    X = pd.read_csv("datasets/onlineRetail.csv")
    X = X[["Quantity", "InvoiceDate", "UnitPrice"]]
    d = 1
    m = 1
    y = 2011
    maxN = len(X["Quantity"])
    for _ in range(t):
        d += 1
        nextDay = X.copy()
        nextDay = (
            nextDay[
                nextDay.InvoiceDate.str.contains(f"{m}/{d}/{y}")
            ]  # american date format...
            .drop("InvoiceDate", axis=1)
            .dropna()
        )
        if len(nextDay["Quantity"]) == 0:
            # skip that day
            continue
        maxN = min(maxN, len(nextDay["Quantity"]))
    return X, maxN - 1


def uber(n=-1):
    f = pd.read_csv("datasets/uber/PreProcessed.csv")
    X1, Y1, Z1, X2, Y2, Z2 = (
        list(f["x1"])[:n],
        list(f["y1"])[:n],
        list(f["z1"])[:n],
        list(f["x2"])[:n],
        list(f["y2"])[:n],
        list(f["z2"])[:n],
    )
    return np.array(list(zip(X1, Y1, Z1))), np.array(list(zip(X2, Y2, Z2)))


def uberSetup1(n=-1):
    X = pd.read_csv("datasets/uber/uber-raw-data-jun14.csv")

    # "Date/Time","Lat","Lon",
    # 6/1/2014 0:00:00
    X = X[["Lon", "Lat", "Date/Time"]]
    nextDay = X.copy()
    nextDay = (
        nextDay[nextDay["Date/Time"].astype(str).str.startswith(f"6/1/2014")]
        .drop("Date/Time", axis=1)
        .dropna()
    )
    R = 6371
    if n == -1:
        n = len(nextDay["Lon"])
    nextDay = nextDay.iloc[:n, :].set_index(np.arange(n))
    nextDay["x"] = np.cos(nextDay["Lat"]) * np.cos(nextDay["Lon"]) * R
    nextDay["y"] = np.cos(nextDay["Lat"]) * np.sin(nextDay["Lon"]) * R
    nextDay["z"] = np.sin(nextDay["Lat"]) * R
    features = ["x", "y", "z"]
    feat = [pd.to_numeric(nextDay[f]) for f in features]
    return np.array(list(zip(*feat)))


def uberGetNext(n, X=pd.DataFrame(), day=1):
    if X.empty:
        X = pd.read_csv("datasets/uber/uber-raw-data-jun14.csv")
        X = X[["Lon", "Lat", "Date/Time"]]
        day = 1
    nextDay = X.copy()
    nextDay = (
        nextDay[nextDay["Date/Time"].astype(str).str.startswith(f"6/{day}/2014")]
        .drop("Date/Time", axis=1)
        .dropna()
    )
    R = 6371
    nextDay = nextDay.iloc[:n, :].set_index(np.arange(n))
    nextDay["x"] = np.cos(nextDay["Lat"]) * np.cos(nextDay["Lon"]) * R
    nextDay["y"] = np.cos(nextDay["Lat"]) * np.sin(nextDay["Lon"]) * R
    nextDay["z"] = np.sin(nextDay["Lat"]) * R
    features = ["x", "y", "z"]
    feat = [pd.to_numeric(nextDay[f]) for f in features]
    return X, day + 1, np.array(list(zip(*feat)))


def uberGetN(t=20):
    X = pd.read_csv("datasets/uber/uber-raw-data-jun14.csv")

    X = X[["Lon", "Lat", "Date/Time"]]
    day = 1
    maxN = len(X["Lon"])
    for _ in range(t):
        nextDay = X.copy()
        nextDay = nextDay[
            nextDay["Date/Time"].astype(str).str.startswith(f"6/{day}/2014")
        ].dropna()
        maxN = min(maxN, len(nextDay["Lon"]))
        day += 1
    return X, maxN - 1
