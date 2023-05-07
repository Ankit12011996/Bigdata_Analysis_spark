import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.functions import col, substring
import pyspark.sql.functions as F
from pyspark.sql import Row
import seaborn as sns


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("question1") \
        .config("spark.local.dir","/fastdata/acr22ar/") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")


#Reading the CSV file into pyspark dataframe.
df = spark.read.format("csv").option("delimiter", " ").load("../Data/NASA_access_log_Jul95").cache()

#Dropping the Column that are uneccessary.
df = df.drop('_c1','_c2','_c4')

#Renaming the column for convenience
cols_names = ['host','timestamp','url','code','count']

#converting to pandas dataframe
df = df.toDF(*cols_names)
df = df.withColumn("date", substring("timestamp", 2, 2)).withColumn("hour", substring("timestamp", 14, 2))

#filtering using the last characters in the hostname.
de_df = df.filter(df.host.endswith(".de"))
ca_df = df.filter(df.host.endswith(".ca"))
sg_df = df.filter(df.host.endswith(".sg"))

#Counting Hosts of countries extracted above germany,canada , singapore
de_count = de_df.select('host').count()
ca_count= ca_df.select('host').count()
sg_count= sg_df.select('host').count()

# Names of countries in a list to make a plot
countries = ['Germany', 'Canada', 'Singapore']

# Counts of countries in a list to make a plot
counts = [de_count, ca_count, sg_count]

plt.bar(countries, counts, color = 'blue', width =0.5)

plt.xlabel('Country')
plt.ylabel('Number of requests')
plt.title('Number of requests per country')
plt.grid(True)
plt.show()
plt.savefig('Q1_figA2.png')

##############################################GERMANY#####################################################################

#calculating frequent visitors in germany and percentage with respect to total visitors.
no_req_de = de_df.select('host').count()
n_host_de = de_df.select('host').distinct().count()
freq_de = de_df.groupBy('host').count().orderBy('count',ascending=False).limit(9)#.show(9,truncate=False)
freq_de = freq_de.withColumn("Percentage", F.col("count") / no_req_de * 100)

# Calculate the remaining count and its percentage
remaining_count_de = no_req_de - freq_de.select(F.sum('count')).collect()[0][0]
remaining_percentage_de = remaining_count_de/ no_req_de * 100

# Create a new Row with these values using Row()
new_row_de = spark.createDataFrame([Row(host="rest", count=int(remaining_count_de), Percentage=float(remaining_percentage_de))])
freq_de = freq_de.union(new_row_de)

print(f"There are {n_host_de} unique hosts FROM Germany")
print(f"The most frequently visited host from Germany is {freq_de}")

#Plotting percentage vs percentage request of frequent host

de_df_pandas = freq_de.toPandas()
# Create a bar graph
plt.figure(figsize=(10,10))
plt.barh(de_df_pandas['host'], de_df_pandas['Percentage'], color='skyblue')
plt.xlabel('Percentage')
plt.ylabel('Host')
plt.title('Percentage of Requests by Host')
plt.gca().invert_yaxis()  # Reverse the order of the hosts

# Save the figure
plt.savefig('Q1_figC1.png')

##############################CANADA############################################################################
#calculating frequent visitors in Canada and percentage with respect to total visitors.

no_req_ca = ca_df.select('host').count()
n_host_ca=ca_df.select('host').distinct().count()
freq_ca = ca_df.groupBy('host').count().orderBy('count',ascending=False).limit(9)#.show(9,truncate=False)
freq_ca = freq_ca.withColumn("Percentage", F.col("count") / no_req_ca * 100)
print(f"There are {n_host_ca} unique hosts FROM Canada")
print(f"The most frequently visited host from Canada is {freq_ca}")

# Calculate the remaining count and its percentage
remaining_count_ca = no_req_ca - freq_ca.select(F.sum('count')).collect()[0][0]
remaining_percentage_ca = remaining_count_ca/ no_req_ca * 100

# Create a new Row with these values using Row()
new_row_ca = spark.createDataFrame([Row(host="rest", count=int(remaining_count_ca), Percentage=float(remaining_percentage_ca))])
freq_ca = freq_ca.union(new_row_ca)

ca_df_pandas = freq_ca.toPandas()
# Create a bar graph
plt.figure(figsize=(10,10))
plt.barh(ca_df_pandas['host'], ca_df_pandas['Percentage'], color='skyblue')
plt.xlabel('Percentage')
plt.ylabel('Host')
plt.title('Percentage of Requests by Host')
plt.gca().invert_yaxis()  # Reverse the order of the hosts

# Save the figure
plt.savefig('Q1_figC2.png')
###########################################Singapore######################################################################

#calculating frequent visitors in Singapore and percentage with respect to total visitors.
no_req_sg = sg_df.select('host').count()
n_host_sg=sg_df.select('host').distinct().count()
freq_sg = sg_df.groupBy('host').count().orderBy('count',ascending=False).limit(9)#.show(9,truncate=False)
freq_sg = freq_sg.withColumn("Percentage", F.col("count") / no_req_sg * 100)
print(f"There are {n_host_sg} unique hosts FROM Singapore")
print(f"The most frequently visited host from Singapore is {freq_sg}")

# Calculate the remaining count and its percentage
remaining_count_sg = no_req_sg - freq_sg.select(F.sum('count')).collect()[0][0]
remaining_percentage_sg = remaining_count_sg/ no_req_sg * 100

# Create a new Row with these values using Row()
new_row_sg = spark.createDataFrame([Row(host="rest", count=int(remaining_count_sg), Percentage=float(remaining_percentage_sg))])
freq_sg = freq_sg.union(new_row_sg)

sg_df_pandas = freq_sg.toPandas()
# Create a bar graph
plt.figure(figsize=(10,10))
plt.barh(sg_df_pandas['host'], sg_df_pandas['Percentage'], color='skyblue')
plt.xlabel('Percentage')
plt.ylabel('Host')
plt.title('Percentage of Requests by Host')
plt.gca().invert_yaxis()  # Reverse the order of the hosts

# Save the figure
plt.savefig('Q1_figC3.png')
######################################################################################################################################

#Creating a 2D array or matrix heatmap_data  to plot graph days vs hour on count of visitors for all the 3 countries germany, canada, singapore.


de_df_host = de_df.groupBy('host').count().orderBy('count',ascending=False).limit(1)
#using collect() to collect the list from a column in pyspark dataFrame.
a=de_df_host.collect()[0][0]
filtered_df = de_df.filter(de_df.host == a)
days = [int(row['date']) for row in filtered_df.collect()]
hours = [int(row['hour']) for row in filtered_df.collect()]
counts = [int(row['count']) if row['count'] != '-' else 0 for row in filtered_df.collect()]
max0= max(days)
heatmap_data = np.zeros((24, max0))
for day, hour, count in zip(days, hours, counts):
    heatmap_data[hour][day-1] = count
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data,cmap = 'coolwarm')
#plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
#plt.colorbar(label='Number of visits')
plt.xlabel('Day')
plt.ylabel('Hour')
plt.title(f'Visits by {a}')
plt.show()
plt.savefig('Q1_figD1.png')

##############################################################################################################################################

ca_df_host = ca_df.groupBy('host').count().orderBy('count',ascending=False).limit(1)

b=ca_df_host.collect()[0][0]
filtered_df_ca = ca_df.filter(ca_df.host == b)
days1 = [int(row['date']) for row in filtered_df_ca.collect()]
hours1 = [int(row['hour']) for row in filtered_df_ca.collect()]
counts1 = [int(row['count']) if row['count'] != '-' else 0 for row in filtered_df_ca.collect()]
max1 = max(days1)

heatmap_data_ca = np.zeros((24, max1))
for day, hour, count in zip(days1, hours1, counts1):
    heatmap_data_ca[hour][day-1] = count 
plt.figure(figsize=(10,6))

sns.heatmap(heatmap_data_ca,cmap = 'coolwarm')

plt.xlabel('Day')
plt.ylabel('Hour')
plt.title(f'Visits by {b}')
plt.show()
plt.savefig('Q1_figD2.png')

######################################################################################################################################

sg_df_host = sg_df.groupBy('host').count().orderBy('count',ascending=False).limit(1)
c=sg_df_host.collect()[0][0]
filtered_df_sg = sg_df.filter(sg_df.host == c)
days2 = [int(row['date']) for row in filtered_df_sg.collect()]
hours2 = [int(row['hour']) for row in filtered_df_sg.collect()]
counts2 = [int(row['count']) if row['count'] != '-' else 0 for row in filtered_df_sg.collect()]
max2 = max(days2)

heatmap_data_sg = np.zeros((24,max2))
for day, hour, count in zip(days2, hours2, counts2):
    heatmap_data_sg[hour][day-1] = count
plt.figure(figsize=(10,6))

sns.heatmap(heatmap_data_sg,cmap = 'coolwarm')
plt.xlabel('Day')
plt.ylabel('Hour')
plt.title(f'Visits by {c}')
plt.show()
plt.savefig('Q1_figD3.png')