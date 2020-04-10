# Cordero Robles, Carlos Alberto
import random as rnd
import data_1_into_2D_list_ as d1il
import data_2_row_types_ as d2rt
import data_3_column_types_ as d3ct

my_file_name = "airfoil_ikb_CACR.csv"

def extract_my_data():
    data_file = "airfoil_self_noise_.csv"
    in_file = open(data_file, 'r')
    #my_file = open("airfoil_ikb_.csv", 'w')
    my_file = open(my_file_name, 'w')

    #rnd.seed(17)
    rnd.seed()
    for line in in_file:
        r = rnd.random()
        #if r < 0.1:
        if r < 0.2:
            my_file.writelines(line)

    in_file.close()
    my_file.close()


def split_train_and_test():
    original_file = open(my_file_name, 'r')
    test_file = open("test.csv", 'w')
    train_file = open("train.csv", 'w')
    trainCnt = 0
    freq = 0
    trainList = []
    #Separate 70% to train and 30% to test
    rnd.seed()
    original_list = original_file.readlines()
    for line in original_list:
        r = rnd.random()
        if (r < 0.7):
            train_file.writelines(line)
        else:
            test_file.writelines(line)
    test_file.close()
    train_file.close()
    original_file.close()
    print("-"*10+"TRAIN"+"-"*10)
    trainList = d1il.make_2D_list("train.csv", 0, 0)
    freq = d1il.get_row_sizes(trainList)
    (freqTypeList, badRowList) = d2rt.row_types(trainList,True)
    d2rt.filter_bad_rows(trainList)
    d2rt.print_row_types()
    d3ct.column_types(trainList)
    d3ct.print_column_types()
    #Create Matrix
    matrixA = []
    vectorY = []
    for row in trainList:
        row.insert(0,1)
        matrixA.append([float(element) for element in row[:-1]])
        vectorY.append([float(element) for element in row[-1:]])
    trasMatrixA = [[matrixA[j][i] for j in range(len(matrixA))] for i in range(len(matrixA[0]))]
    # Make A a square matrix
    matrixA = mulMat(trasMatrixA,matrixA)
    vectorY = mulMat(trasMatrixA, vectorY)

    # Create the system and solve
    systemMatrix = list(matrixA)
    x = []
    for row in range(len(systemMatrix)):
        systemMatrix[row].append(vectorY[row][0])
    gaussJordan(systemMatrix)
    for xVal in range(0, len(systemMatrix)):
        print("The value of the b"+ str(xVal), " is ", systemMatrix[xVal][-1])
        auxList = [systemMatrix[xVal][-1]]
        x.append(auxList)

    # TEST
    matrixATest = []
    vectorYTest = []
    print("-" * 10 + "TEST" + "-" * 10)
    testList = d1il.make_2D_list("test.csv", 0, 0)
    freqTest = d1il.get_row_sizes(testList)
    (freqTypeListTest, badRowListTest) = d2rt.row_types(testList,True)
    d2rt.filter_bad_rows(testList)
    d2rt.print_row_types()
    d3ct.column_types(testList)
    d3ct.print_column_types()
    for row in testList:
        row.insert(0,1)
        matrixATest.append([float(element) for element in row[:-1]])
        vectorYTest.append([float(element) for element in row[-1:]])
    # Optain yPrima
    yPrima = mulMat(matrixATest, x)
    # Calculate RMSE
    auxSquare = 0
    RMSE = 0
    for output in range(0, len(vectorYTest)):
        auxSquare += ((yPrima[output][0] - vectorYTest[output][0]) ** 2)
    RMSE = (auxSquare / len(vectorYTest)) ** (1/2)
    print("#"*40)
    print("The value of the RMSE", " is ", RMSE)
    print("#"*40)

def mulMat(matrixA, matrixB):
    result = []
    for row in range(0, len(matrixA)):
        result.append([0] * len(matrixB[0]))
    for i in range(len(matrixA)):
        for j in range(len(matrixB[0])):
            for k in range(len(matrixB)):
                result[i][j] += float(matrixA[i][k]) * float(matrixB[k][j])
    return result

def gaussJordan(matrix):
  (rows, columns) = (len(matrix), len(matrix[0]))
  almostCero = 1.0/(10**10)
  for row in range(0,rows):
    maxrow = row
    # Find max pivot
    for auxRow in range(row+1, rows):
      if abs(matrix[auxRow][row]) > abs(matrix[maxrow][row]):
        maxrow = auxRow
    # Swap current row with the max current Pivot
    (matrix[row], matrix[maxrow]) = (matrix[maxrow], matrix[row])
    if abs(matrix[row][row]) <= almostCero:
      # Not possible to solve
      return False
    # Eliminate column row
    for auxRow in range(row+1, rows):
      c = matrix[auxRow][row] / matrix[row][row]
      for column in range(row, columns):
        matrix[auxRow][column] -= matrix[row][column] * c
  # Backsubstitute
  for row in range(rows-1, 0-1, -1):
    c  = matrix[row][row]
    for auxRow in range(0,row):
      for column in range(columns-1, row-1, -1):
        matrix[auxRow][column] -=  matrix[row][column] * matrix[auxRow][row] / c
    matrix[row][row] /= c
    # Normalize row row
    for column in range(rows, columns):
      matrix[row][column] /= c
  return True

if __name__ == "__main__":
    #extract_my_data()
    split_train_and_test()

