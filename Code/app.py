import sys
from math import radians, acos, cos
from pyspark.sql import SparkSession
import pyspark.sql.functions as sparkFun
from pyspark.sql.types import DoubleType, StringType
import time


# curried function that passes "target" parameter to dist udf function

def make_udf_function(target):

    # function dist calculates the distance between loc_a and loc_b

    # Parameters
    # -------------
    # vehicle_x: Column(longitude)
    # vehicle_y: Column(latitiude)
    # target: tuple(number, number) -> point of interest


    # Returns
    # -------------
    # number -> distance between pairs of type ("vehicle location", target) datapoints

    def dist(vehicles_x, vehicles_y, target=target, r=6371):

        coordinates = vehicles_y, vehicles_x, target[1], target[0]

        phi1, lambda1, phi2, lambda2 = [
            radians(c) for c in coordinates
        ]
        
        # apply the Haversine formula
        dist = r * acos(cos(phi2 - phi1) - cos(phi1) * cos(phi2) * (1-cos(lambda2 - lambda1)))
        return dist
    
    return sparkFun.udf(dist, DoubleType())



# function for extarcting id from lane name

def parse_lane_name(lane_name):
    # Parameters
    # -------------
    # lane_name: String -> lane_name in form "lane_id"_"lane_num" or "lane_id"#"lane_num"

    # Returns
    # -------------
    # lane_id: String
    lane_name = lane_name.replace(":", "").replace("-", "")
    if "#" in lane_name:
        return lane_name.split("#")[0]
    else:
        return lane_name.split("_")[0]



def make_cut_timestep_udf(time_window_size):
    def cut_timestep(timestep_value, window_size=time_window_size):
        return timestep_value // window_size
    return sparkFun.udf(cut_timestep)



# main

def main(args):

    spark = SparkSession.builder.appName("SmartCityMobility") \
                                .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # input_path = "./input/emission_data.csv"
    # vehicle_type = None
    # poi_geo_coords = (23.728415, 37.983736)
    # max_dist = 0.3
    # time_window = (50., 100.)
    # time_window_size = 500.

    # parse input arguments
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    poi_geo_coords = (float(sys.argv[3]), float(sys.argv[4]))
    max_dist = float(sys.argv[5])
    time_window = (float(sys.argv[6]), float(sys.argv[7]))
    vehicle_type = sys.argv[8] if len(sys.argv) == 9 else None
    time_window_size = 500.

    #read data from csv
    data = spark.read.option("header", True).option("inferSchema", True).csv(input_path, sep=";")

    # check data
    data.show(10)
    data.printSchema()

    #removing null rows
    data = data.filter(data.vehicle_id.isNotNull())

    # creating user defined function for filtering data based on geo-coords
    dist_fun = make_udf_function(poi_geo_coords)

    # filtering the initial dataset containing all vehicles
    # saving filtered result to memory as it will be required by subsequent actions
    vehiclesInTimeWindow = data.filter((data.timestep_time >= time_window[0]) & (data.timestep_time <= time_window[1]))

    # TASK 1 -----------------------------------------------

    closeVehicles = vehiclesInTimeWindow.filter(dist_fun(data.vehicle_x, data.vehicle_y) <= max_dist)

    if vehicle_type:
        closeVehicles = closeVehicles.filter(data.vehicle_type == vehicle_type)
    
    closeVehicles.cache()

    print(f"Number of vehicles close to {poi_geo_coords}: {closeVehicles.count()}")
    closeVehicles.show()
    closeVehicles.write.format("csv").mode("overwrite").option("header", True).save(f"{output_path}/close_vehicles")

    # ENDOF TASK 1 -----------------------------------------




    # TASK 2 -----------------------------------------------

    timestep_udf = make_cut_timestep_udf(time_window_size)
    data = data.withColumn("time_period", timestep_udf(data.timestep_time))

    lane_parsed = sparkFun.udf(parse_lane_name, StringType())

    grpouped = data.groupBy("time_period", lane_parsed(data.vehicle_lane))
    grpouped.count().show()
    
    stat_co2 = grpouped.agg(sparkFun.min("vehicle_CO2"), sparkFun.max("vehicle_CO2"), sparkFun.avg("vehicle_CO2"), sparkFun.stddev("vehicle_CO2"))
    stat_co = grpouped.agg(sparkFun.min("vehicle_CO"), sparkFun.max("vehicle_CO"), sparkFun.avg("vehicle_CO"), sparkFun.stddev("vehicle_CO"))
    stat_hc = grpouped.agg(sparkFun.min("vehicle_HC"), sparkFun.max("vehicle_HC"), sparkFun.avg("vehicle_HC"), sparkFun.stddev("vehicle_HC"))
    stat_pmx = grpouped.agg(sparkFun.min("vehicle_PMx"), sparkFun.max("vehicle_PMx"), sparkFun.avg("vehicle_PMx"), sparkFun.stddev("vehicle_PMx"))
    stat_nox = grpouped.agg(sparkFun.min("vehicle_NOx"), sparkFun.max("vehicle_NOx"), sparkFun.avg("vehicle_NOx"), sparkFun.stddev("vehicle_NOx"))
    stat_fuel = grpouped.agg(sparkFun.min("vehicle_fuel"), sparkFun.max("vehicle_fuel"), sparkFun.avg("vehicle_fuel"), sparkFun.stddev("vehicle_fuel"))

    stat_co2.show()
    stat_co.show()
    stat_hc.show()
    stat_pmx.show()
    stat_nox.show()
    stat_fuel.show()

    stat_co2.write.format("csv").mode("overwrite").option("header", True).save(f"{output_path}/stat_co2")
    stat_co.write.format("csv").mode("overwrite").option("header", True).save(f"{output_path}/stat_co")
    stat_hc.write.format("csv").mode("overwrite").option("header", True).save(f"{output_path}/stat_hc")
    stat_pmx.write.format("csv").mode("overwrite").option("header", True).save(f"{output_path}/stat_pmx")
    stat_nox.write.format("csv").mode("overwrite").option("header", True).save(f"{output_path}/stat_nox")
    stat_fuel.write.format("csv").mode("overwrite").option("header", True).save(f"{output_path}/stat_fuel")

    # ENDOF TASK 2 -----------------------------------------


    spark.stop()




if __name__ == "__main__":
    if(len(sys.argv) < 8):
        print("Missing parameters...")
        exit(-1)

    start = time.time()
    main(sys.argv)
    end = time.time()

    print(f"Execution time: {end - start} s.")