import unittest
from train import main
import os

if "TRAINDATA_PATH" not in os.environ.keys():
    raise ValueError("The unittests requires the environment variables 'TRAINDATA_PATH' and 'VALIDATA_PATH' to be set")

class TestTraining(unittest.TestCase):

    def get_common_args(self):
        return dict(
            batch_size=1,
            nworkers=0,
            outdir="/tmp/",
            num_epochs=1,
            snapshot=None,
            finetune=None,
            lr=0.01,
            lradapt=1,
            labelimage="buildings10m.tif",
            smoketest=True,
            trainpath=os.environ["TRAINDATA_PATH"],
            validpath=os.environ["VALIDATA_PATH"])

    def test_experiment_s1(self):

        args = self.get_common_args()
        args["experiment"] = "s1"
        try:
            main(**args)
        except Exception as err:
            raise err
            self.fail(err)

    def test_experiment_s2(self):

        args = self.get_common_args()
        args["experiment"] = "s2"
        try:
            main(**args)
        except Exception as err:
            self.fail(err)

    def test_experiment_vhr(self):

        args = self.get_common_args()
        args["experiment"] = "vhr"
        try:
            main(**args)
        except Exception as err:
            self.fail(err)

    def test_experiment_vhrs1(self):

        args = self.get_common_args()
        args["experiment"] = "vhrs1"
        try:
            main(**args)
        except Exception as err:
            self.fail(err)


    def test_experiment_vhrs2(self):

        args = self.get_common_args()
        args["experiment"] = "vhrs2"
        try:
            main(**args)
        except Exception as err:
            self.fail(err)


    def test_experiment_s1s2(self):

        args = self.get_common_args()
        args["experiment"] = "s1s2"
        try:
            main(**args)
        except Exception as err:
            self.fail(err)

    def test_experiment_vhrs1s2(self):

        args = self.get_common_args()
        args["experiment"] = "vhrs1s2"
        try:
            main(**args)
        except Exception as err:
            self.fail(err)

if __name__ == '__main__':
    unittest.main()