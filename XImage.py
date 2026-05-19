from osgeo import gdal, gdal_array
import numpy as np


def GetSuffix(filepath):
    i = filepath.rfind('.')
    str = ""
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i + 1].isdigit() or len(filepath) - i > 5:
        return str
    else:
        str = filepath[i + 1:len(filepath)]
        return str


def GetFilePath(filepath):
    i = filepath.rfind('.')
    str = ""
    if i == -1 or i == 0 or (i == 1 and filepath[0] == '.') or filepath[i + 1].isdigit():
        return filepath
    else:
        str = filepath[0:i]
        return str


class CXImage():
    m_nBands = 0
    m_nLines = 0
    m_nSamples = 0
    m_nClasses = 0

    m_nDataType = np.uint8
    m_strImgPath = ""

    currentHeight = 0
    currentWidth = 0
    partWidth = 0
    partHeight = 0
    isNormalization = True

    proDataset = None
    currentPosX = 0
    currentPosY = 0

    minBandValue = None
    maxBandValue = None

    data_arrange = 0

    def __init__(self, nBands=0, nLines=0, nSamples=0, nDataType=np.uint8, nClass=0, strImgPath=None, data_arrange=0):
        self.m_nBands = nBands
        self.m_nLines = nLines
        self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_nClasses = nClass
        self.m_strImgPath = strImgPath
        self.data_arrange = data_arrange

        self.minBandValue = None
        self.maxBandValue = None
        self.isNormalization = False
        self.proDataset = None

    def Create(self, nBands, nLines, nSamples, nDataType, strImgPath, nClass=0, options=None):
        nDataType_result = gdal_array.NumericTypeCodeToGDALTypeCode(nDataType)

        self.m_nBands = nBands
        self.m_nLines = nLines
        self.m_nSamples = nSamples
        self.m_nDataType = nDataType
        self.m_strImgPath = strImgPath
        self.m_nClasses = nClass

        self.partWidth = self.m_nSamples
        self.partHeight = self.m_nLines
        self.currentPosX = 0
        self.currentPosY = 0

        self.initNextWidowsSize()

        suffix = GetSuffix(self.m_strImgPath)
        if suffix == "bmp":
            driver = gdal.GetDriverByName("BMP")
        elif suffix == "jpg":
            driver = gdal.GetDriverByName("JPEG")
        elif suffix == "tif" or suffix == "tiff":
            driver = gdal.GetDriverByName("GTiff")
        elif suffix == "bt":
            driver = gdal.GetDriverByName("BT")
        elif suffix == "ecw":
            driver = gdal.GetDriverByName("ECW")
        elif suffix == "fits":
            driver = gdal.GetDriverByName("FITS")
        elif suffix == "gif":
            driver = gdal.GetDriverByName("GIF")
        elif suffix == "hdf":
            driver = gdal.GetDriverByName("HDF4")
        elif suffix == "hdr":
            driver = gdal.GetDriverByName("EHdr")
        elif suffix == "" or suffix == "img":
            driver = gdal.GetDriverByName("ENVI")
        else:
            print("GetDriverByName Error!")
            return

        if options is None:
            self.proDataset = driver.Create(self.m_strImgPath, nSamples, nLines, self.m_nBands,
                                        nDataType_result)
        else:
            self.proDataset = driver.Create(self.m_strImgPath, nSamples, nLines, self.m_nBands,
                                            nDataType_result, options)

    def Open(self, strImgPath):
        self.m_strImgPath = strImgPath

        self.proDataset = gdal.Open(self.m_strImgPath)
        if self.proDataset == None:
            print("file open error")
            return False
        self.m_nBands = self.proDataset.RasterCount
        self.m_nLines = self.proDataset.RasterYSize
        self.m_nSamples = self.proDataset.RasterXSize

        self.partWidth = self.m_nSamples
        self.partHeight = self.m_nLines
        self.currentPosX = 0
        self.currentPosY = 0
        self.currentHeight = self.m_nLines
        self.currentWidth = self.m_nSamples

        self.initNextWidowsSize()
        return True

    def next(self, padding=4):
        self.currentPosX += self.currentWidth - padding
        self.currentPosY += 0

        if self.currentPosX + padding >= self.m_nSamples:
            self.currentPosX = 0
            self.currentPosY += self.currentHeight - padding
        if self.currentPosY + padding >= self.m_nLines:
            self.currentPosX = 0
            self.currentPosY = 0
            return False

        self.initNextWidowsSize()
        return True

    def setPartSize(self, setPartWidth=-1, setPartHeight=-1, memorySize=-1, typeSize=4, safetyFactor=6):
        if setPartWidth != -1 and setPartHeight != -1:
            self.partHeight = setPartHeight
            self.partWidth = setPartWidth
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        elif setPartWidth != -1 and setPartHeight == -1:
            ratio = float(self.m_nLines) / self.m_nSamples
            self.partWidth = setPartWidth
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
            return
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize != -1:
            if memorySize < 1000:
                self.partWidth = 0
                self.partHeight = 0
                return
            ratio = float(self.m_nLines) / self.m_nSamples
            self.partWidth = int((memorySize * 1000.0 / (typeSize * ratio * safetyFactor)) ** 0.5)
            self.partHeight = int(self.partWidth * ratio)
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        elif setPartWidth == -1 and setPartHeight == -1 and memorySize == -1:
            self.partHeight = self.m_nLines
            self.partWidth = self.m_nSamples
            self.currentPosX = 0
            self.currentPosY = 0
            self.initNextWidowsSize()
        else:
            print("setPartSize parameters error!")
            return

    def setClassNo(self, classNo):
        self.m_nClasses = classNo
        if self.m_nBands == 1 and self.m_nClasses > 0:
            self.setClassificationColor()
        return

    def setClassificationColor(self, trainFile=""):
        isUsedDefault = False
        if trainFile == "":
            isUsedDefault = True
        Is = open(trainFile, "r")
        if Is == None:
            print("Open ROI file Error!")
            isUsedDefault = True
        pIn = ""
        if not isUsedDefault:
            pIn = Is.readline()
            pIn = Is.readline()
            self.m_nClasses = pIn.split()[4]
            pIn = Is.readline()

        usedColor = np.zeros(3 * (int(self.m_nClasses) + 1), dtype=int)
        pColor = [0, 0, 0,   0, 0, 255,   46, 139, 87,   0, 255, 0,   216, 191, 216,   255, 0, 0,   255, 255, 255,
                   255, 255, 0,   0, 255, 255,   255, 0, 255,   48, 48, 48,   128, 0, 0,   0, 128, 0,   0, 0, 128,
                  128, 128, 0,   0, 128, 128,   128, 0, 128,   255, 128, 0,   128, 255, 0,   255, 0, 128]

        colorTable = gdal.ColorTable(gdal.GPI_RGB)
        pBand = self.proDataset.GetRasterBand(1)

        for i in range(int(self.m_nClasses) + 1):
            if isUsedDefault or i == 0:
                usedColor[3 * i] = pColor[(3 * i) % 60]
                usedColor[3 * i + 1] = pColor[(3 * i + 1) % 60]
                usedColor[3 * i + 2] = pColor[(3 * i + 2) % 60]
            else:
                pIn = Is.readline()
                pIn = Is.readline()
                pIn = Is.readline()
                strTemp = pIn.split()[4]
                str = strTemp[1:len(strTemp) - 1]
                usedColor[3 * i] = int(str)

                strTemp = pIn.split()[5]
                str = strTemp[0:1]
                usedColor[3 * i + 1] = int(str)

                strTemp = pIn.split()[6]
                str = strTemp[0:1]
                usedColor[3 * i + 2] = int(str)

                pIn = Is.readline()
            colorEntry = gdal.ColorEntry()
            colorEntry.c1 = usedColor[3 * i]
            colorEntry.c2 = usedColor[3 * i + 1]
            colorEntry.c3 = usedColor[3 * i + 2]
            colorEntry.c4 = 0
            colorTable.SetColorEntry(i, colorEntry)

        return pBand.SetColorTable(colorTable)

    def getPartionNum(self, padding=10):
        temp_currentPosX = self.currentPosX
        temp_currentPosY = self.currentPosY
        temp_currentHeight = self.currentHeight
        temp_currentWidth = self.currentWidth

        self.currentPosX = 0
        self.currentPosY = 0
        self.initNextWidowsSize()
        partionNum = 1
        maxPartionWidth = 0
        maxPartionHeight = 0

        while True:
            if maxPartionHeight < self.currentHeight:
                maxPartionHeight = self.currentHeight
            if maxPartionWidth < self.currentWidth:
                maxPartionWidth = self.currentWidth
            if not self.next(padding):
                break
            partionNum += 1
        self.currentPosX = temp_currentPosX
        self.currentPosY = temp_currentPosY
        self.currentHeight = temp_currentHeight
        self.currentWidth = temp_currentWidth
        return partionNum, maxPartionHeight, maxPartionWidth

    def setPartionParameters(self, currentPosXValue=-1, currentPosYValue=-1, currentWidthValue=-1,
                             currentHeightValue=-1):
        if currentHeightValue > 0 and currentPosXValue > -1 and currentPosYValue > -1 and currentWidthValue > 0:
            self.currentHeight = currentHeightValue
            self.currentWidth = currentWidthValue
            self.currentPosX = currentPosXValue
            self.currentPosY = currentPosYValue

    def initMinMaxBandValue(self, is2PercentScale=False, percentValue=0.02):
        return

    def uninitMinMaxBandValue(self):
        return

    def setNormalizaion(self, setIsNormalization=True):
        self.isNormalization = setIsNormalization
        return

    def setHeaderInformation(self, img):
        projectionRef = img.proDataset.GetProjectionRef()
        self.proDataset.SetProjection(projectionRef)

        padfTransform = img.proDataset.GetGeoTransform()
        self.proDataset.SetGeoTransform(padfTransform)

    def GetData(self, dataType, cHeight=-1, cWidth=-1, cPosX=-1, cPosY=-1):
        if cHeight == -1 and cWidth == -1 and cPosX == -1 and cPosY == -1:

            data = self.proDataset.ReadAsArray(self.currentPosX, self.currentPosY, self.currentWidth,
                                               self.currentHeight)

            if data.ndim == 3 and self.data_arrange == 0:
                data = data.transpose(1, 2, 0)
            elif data.ndim == 3 and self.data_arrange == 1:
                pass
            elif data.ndim == 2:
                pass
            else:
                raise AttributeError("data shape error!")

            data = data.astype(dataType)

            return data
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:

            data = self.proDataset.ReadAsArray(cPosX, cPosY, cWidth, cHeight)
            if data.ndim == 3 and self.data_arrange == 0:
                data = data.transpose(1, 2, 0)
            elif data.ndim == 3 and self.data_arrange == 1:
                pass
            elif data.ndim == 2:
                pass
            else:
                raise AttributeError("data shape error!")

            data = data.astype(dataType)

            return data
        else:
            raise AttributeError("GetData parameters error!")

    def NormlizedData(self, data, setMin=0, setMax=1):
        return

    def WriteImgData(self, pData, cHeight=-1, cWidth=-1, cPosX=0, cPosY=0, padding=4):
        if (cHeight == self.m_nLines or cHeight == -1) and (cWidth == self.m_nSamples or cWidth == -1) and cPosX == 0 and cPosY == 0:
            temp_currentPosX = self.currentPosX
            temp_currentPosY = self.currentPosY
            temp_currentWidth = self.currentWidth
            temp_currentHeight = self.currentHeight
            if self.currentPosX > 0:
                temp_currentPosX = self.currentPosX + padding / 2
                temp_currentWidth = self.currentWidth - padding / 2
            if self.currentPosY > 0:
                temp_currentPosY = self.currentPosY + padding / 2
                temp_currentHeight = self.currentHeight - padding / 2
            if self.currentPosX + self.currentWidth < self.m_nSamples:
                temp_currentWidth = temp_currentWidth - padding / 2
            if self.currentPosY + self.currentHeight < self.m_nLines:
                temp_currentHeight = temp_currentHeight - padding / 2

            if self.data_arrange == 0 and pData.ndim == 3:
                if self.currentPosX == 0 and self.currentPosY == 0:
                    temp_data = pData[:int(temp_currentHeight), :int(temp_currentWidth), :]
                elif self.currentPosX > 0 and self.currentPosY == 0:
                    temp_data = pData[:int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2), :]
                elif self.currentPosX > 0 and self.currentPosY > 0:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2), :]
                else:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth), :]
                temp_data = temp_data.transpose(2, 0, 1)
            elif self.data_arrange == 1 and pData.ndim == 3:
                if self.currentPosX == 0 and self.currentPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight), :int(temp_currentWidth)]
                elif self.currentPosX > 0 and self.currentPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                elif self.currentPosX > 0 and self.currentPosY > 0:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                else:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth)]
                temp_data = temp_data.transpose(2, 0, 1)
            elif self.data_arrange == 0 and pData.ndim == 2:
                if self.currentPosX == 0 and self.currentPosY == 0:
                    temp_data = pData[:int(temp_currentHeight), :int(temp_currentWidth)]
                elif self.currentPosX > 0 and self.currentPosY == 0:
                    temp_data = pData[:int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                elif self.currentPosX > 0 and self.currentPosY > 0:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                else:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth)]
            elif self.data_arrange == 1 and pData.ndim == 2:
                if self.currentPosX == 0 and self.currentPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight), :int(temp_currentWidth)]
                elif self.currentPosX > 0 and self.currentPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                elif self.currentPosX > 0 and self.currentPosY > 0:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                else:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth)]
            else:
                raise AttributeError("data_arrange or pData.ndim error!")

            temp_data = temp_data.astype(self.m_nDataType)

            self.proDataset.WriteRaster(int(temp_currentPosX), int(temp_currentPosY), int(temp_currentWidth), int(temp_currentHeight),
                                        temp_data.tobytes())
            self.proDataset.FlushCache()

            if not np.all(temp_data == 0):
                del temp_data
            return True
        elif cHeight != -1 and cWidth != -1 and cPosX != -1 and cPosY != -1:
            temp_currentPosX = cPosX
            temp_currentPosY = cPosY
            temp_currentWidth = cWidth
            temp_currentHeight = cHeight
            if cPosX > 0:
                temp_currentPosX = cPosX + padding / 2
                temp_currentWidth = cWidth - padding / 2
            if cPosY > 0:
                temp_currentPosY = cPosY + padding / 2
                temp_currentHeight = cHeight - padding / 2
            if cPosX + cWidth < self.m_nSamples:
                temp_currentWidth = temp_currentWidth - padding / 2
            if cPosY + cHeight < self.m_nLines:
                temp_currentHeight = temp_currentHeight - padding / 2

            if self.data_arrange == 0 and pData.ndim == 3:
                if cPosX == 0 and cPosY == 0:
                    temp_data = pData[:int(temp_currentHeight), :int(temp_currentWidth), :]
                elif cPosX > 0 and cPosY == 0:
                    temp_data = pData[:int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2), :]
                elif cPosX > 0 and cPosY > 0:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2), :]
                else:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth), :]
                temp_data = temp_data.transpose(2, 0, 1)
            elif self.data_arrange == 1 and pData.ndim == 3:
                if cPosX == 0 and cPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight), :int(temp_currentWidth)]
                elif cPosX > 0 and cPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                elif cPosX > 0 and cPosY > 0:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                else:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth)]
                temp_data = temp_data.transpose(2, 0, 1)
            elif self.data_arrange == 0 and pData.ndim == 2:
                if cPosX == 0 and cPosY == 0:
                    temp_data = pData[:int(temp_currentHeight), :int(temp_currentWidth)]
                elif cPosX > 0 and cPosY == 0:
                    temp_data = pData[:int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                elif cPosX > 0 and cPosY > 0:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                else:
                    temp_data = pData[int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth)]
            elif self.data_arrange == 1 and pData.ndim == 2:
                if cPosX == 0 and cPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight), :int(temp_currentWidth)]
                elif cPosX > 0 and cPosY == 0:
                    temp_data = pData[:, :int(temp_currentHeight),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                elif cPosX > 0 and cPosY > 0:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                int(padding / 2):int(temp_currentWidth + padding / 2)]
                else:
                    temp_data = pData[:, int(padding / 2):int(temp_currentHeight + padding / 2),
                                :int(temp_currentWidth)]
            else:
                raise AttributeError("data_arrange or pData.ndim error!")

            temp_data = temp_data.astype(self.m_nDataType)

            self.proDataset.WriteRaster(int(temp_currentPosX), int(temp_currentPosY), int(temp_currentWidth),
                                        int(temp_currentHeight), temp_data.tobytes())
            self.proDataset.FlushCache()

            if not np.all(temp_data == 0):
                del temp_data
            return True
        else:
            raise AttributeError("WriteImgData parameters error!")

    def initNextWidowsSize(self):
        acceptWidth = 0.5 * self.partWidth
        acceptHeight = 0.5 * self.partHeight

        self.currentWidth = self.partWidth
        self.currentHeight = self.partHeight

        if self.m_nSamples <= self.currentWidth:
            self.currentWidth = self.m_nSamples
        if self.m_nLines <= self.currentHeight:
            self.currentHeight = self.m_nLines

        if self.currentPosX + self.partWidth > self.m_nSamples or acceptWidth + self.currentPosX + self.partWidth > self.m_nSamples:
            self.currentWidth = self.m_nSamples - self.currentPosX
        if self.currentPosY + self.partHeight > self.m_nLines or acceptHeight + self.currentPosY + self.partHeight > self.m_nLines:
            self.currentHeight = self.m_nLines - self.currentPosY


if __name__ == '__main__':
    pass