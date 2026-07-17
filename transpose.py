import numpy as np
def create_matrix(mc):
    print("\n Array "+str(mc)+"Eement:")
    array_1=map(int,input().split())
    array_1=np.array(list(array_1))
    print("\nArray"+str(mc)+"Row$ Column:")
    row,column=map(int,input().split())
    if (len(array_1)!=(row*column)):
        print("\nRow and column size not match with total elements..retry")
        return create_matrix(mc)
    array_1=array_1.reshape(row,column)
    print("\nArray"+str(mc))
    print(array_1)
    print("\n Transpose")
    return array_1
print(create_matrix(1).transpose())

           