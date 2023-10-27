import os
from pprint import pprint
import yaml
from SelectiveMultiScalewithWraparound2D import *
from TestEnvironmentPathPlanning import *

profile_value = os.environ.get("Profile", "default")
profile_file = Path(f"./Scripts/experiment.{profile_value}.yml")


config=yaml.safe_load(open(profile_file,'r'))




def SelectiveMultiScale(Cities=['Newyork'], index=0, configs_file="Datasets/profile.yml",run=False, plotting=False):
    configs=yaml.load(open(configs_file,'r'),Loader=yaml.FullLoader)
    configs = configs["SelectiveMultiScale"]
    for City in Cities:
        scaleType = "Single"
        runningAllPathsFromCity(City, scaleType, configs, run=run, plotting=True)
        print("")
        scaleType = "Multi"
        runningAllPathsFromCity(City, scaleType, configs, run=run, plotting=True)

        """ Multi versus Single over Large Velocity Range"""
        multiVsSingle(City, index,configs, 500, run=run, plotting=True)
        CumalativeError_SinglevsMulti(City, index, configs, run=run, plotting=True)
        plotMultiplePathsErrorDistribution(City, configs, run=run, plotting=True)
        
        if City == 'Kitti':
            plotKittiGT_singlevsMulti(0)

def main():
    exps=config["Experiments"]
    for k,v in exps.items():
        # run function by k with args and kargs in v
        if not v['run']:
            continue
        func=globals()[k]
        if v['args'] is None:
            func(**v['kwargs'])
        else:
            func(*v['args'],**v['kwargs'])

if __name__== "__main__":  
    main()
    