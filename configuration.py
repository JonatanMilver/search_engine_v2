import os


class ConfigClass:
    # def __init__(self, corpus_path='', output_path='', stemming=False):
    def __init__(self):
        # self.corpusPath = corpus_path
        # self.corpusPath = "C:\\Users\\yonym\\Desktop\\ThirdYear\\IR\\engineV1\\Data\\date=07-27-2020\\covid19_07-27.snappy.parquet"
        self.corpusPath = r"C:\Users\Guyza\OneDrive\Desktop\Information_Systems\University\Third_year\Semester_E\Information_Retrieval\Search_Engine_Project\Data\Data\date=07-27-2020\covid19_07-27.snappy.parquet"

        # try:
        #     os.mkdir(output_path)
        # except:
        #     pass
        # self.savedFileMainFolder = output_path

        # link to a zip file in google drive with your pretrained model
        self._model_url = None
        # False/True flag indicating whether the testing system will download
        # and overwrite the existing model files. In other words, keep this as
        # False until you update the model, submit with True to download
        # the updated model (with a valid model_url), then turn back to False
        # in subsequent submissions to avoid the slow downloading of the large
        # model file with every submission.
        self._download_model = False

        # self.corpusPath = ''
        self.savedFileMainFolder = ''
        self.saveFilesWithStem = self.savedFileMainFolder + "/WithStem"
        self.saveFilesWithoutStem = self.savedFileMainFolder + "/WithoutStem"
        self.toStem = False

        print('Project was created successfully..')



    def get__corpusPath(self):
        return self.corpusPath

    def get_out_path(self):
        return self.savedFileMainFolder

    def get_model_url(self):
        return self._model_url

    def get_download_model(self):
        return self._download_model
