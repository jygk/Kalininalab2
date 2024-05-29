from PIL import Image
import numpy as np
def rgbtoycbcr(ar):
    y = np.zeros(len(ar)*len(ar[0]))
    Cb = np.zeros(len(ar)*len(ar[0]))
    Cr = np.zeros(len(ar)*len(ar[0]))
    for i in range(len(ar)):
        for j in range(len(ar[0])):
            y[j+i*len(ar[0])] = ar[i][j][0] * 0.299 + 0.587 * ar[i][j][1] + 0.114 * ar[i][j][2] + 16
            Cb[j+i*len(ar[0])] = ar[i][j][0] * (-0.148) - 0.29 * ar[i][j][1] + 0.439 * ar[i][j][2] + 128
            Cr[j+i*len(ar[0])] = ar[i][j][0] * 0.439 - 0.367 * ar[i][j][1] + 0.07 * ar[i][j][2] + 128
    return y,Cb,Cr
def downsamplingdel(ar):
    delstr = []
    for i in range(len(ar)):
        if (i%2==0):
            delstr.append(i)
    newarr = np.delete(ar,delstr,0)
    newarr = np.delete(newarr, delstr,1)
    return newarr
def downsamplingapprox(ar):
    new = np.zeros(int(len(ar)/2)*int(len(ar[0])/2))
    new = np.reshape(new,(int(len(ar)/2),int(len(ar[0])/2)))
    for i in range(len(new)):
        for j in range(len(new[0])):
            new[i][j]=(ar[2*i][2*j]+ar[2*i+1][2*j]+ar[2*i][2*j+1]+ar[2*i+1][2*j+1])/4
    return new
def amsampling(ar):
    new = np.zeros(len(ar)*2*len(ar[0])*2)
    new = np.reshape(new,(len(ar)*2,len(ar[0])*2))
    for i in range(len(new)):
        for j in range(len(new[0])):
            new[i][j] = ar[i//2][j//2]
    return new
def diag(ar):
    s = []
    i=0
    j=0
    k = 0
    while(i!=len(ar))and(j!=len(ar[0])):
        while(k%2==0):
            s.append(ar[i][j])
            if (i == (len(ar)-1)):
                j+=1
                break
            if (j==0):
                i = k + 1
                break
            i+=1
            j-=1
        while (k % 2 != 0):
            s.append(ar[i][j])
            if (j == (len(ar[0])-1)):
                i+=1
                break
            if (i == 0):
                j = k + 1
                break
            i -= 1
            j += 1
        k+=1
    return s

def matri(s,l,n):
    ar = np.zeros((l,n))
    diag = 0
    k = 0
    i=j=0
    while (i != l) and (j != n):
        while (diag % 2 == 0):
            ar[i][j] = s[k]
            if (i == (l - 1)):
                j += 1
                k+=1
                break
            if (j == 0):
                i = diag + 1
                k+=1
                break
            i += 1
            j -= 1
            k+=1
        while (diag % 2 != 0):
            ar[i][j] = s[k]
            if (j == (n - 1)):
                i += 1
                k+=1
                break
            if (i == 0):
                j = diag + 1
                k+=1
                break
            i -= 1
            j += 1
            k+=1
        diag += 1
    return np.int32(ar)
def ycbcrtorgb(ar):
    r = np.zeros(len(ar)*len(ar[0])*3)
    r = np.reshape(r,(len(ar),len(ar[0]),3))
    for i in range(len(ar)):
        for j in range(len(ar[0])):
            # r[j+i*len(ar[0])] = (ar[i][j][0]-16) * 0.774 - 0.456 * (ar[i][j][1]-128) + 1.597 * (ar[i][j][2]-128)
            # g[j+i*len(ar[0])] = (ar[i][j][0]-16) * 1.116 - 0.16 * (ar[i][j][1]-128) - 0.814 * (ar[i][j][2]-128)
            # b[j+i*len(ar[0])] = (ar[i][j][0]-16) * 0.998 + 2.019 * (ar[i][j][1]-128) + 9.12 * (ar[i][j][2]-128)
            r[i][j][0] = (ar[i][j][0]-16) * 0.774 - 0.456 * (ar[i][j][1]-128) + 1.597 * (ar[i][j][2]-128)+10
            r[i][j][1] = ar[i][j][0] - 0.344 * (ar[i][j][1] - 128) - 0.714 * (ar[i][j][2] - 128)
            r[i][j][2] = ar[i][j][0] + 1.772 * (ar[i][j][1] - 128) + 0.00000041 * (ar[i][j][2] - 128)
    return r

def DCT(ar):
    n = 8
    global M
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (i==0):
                m[i][j]=np.sqrt(1/n)
            else:
                m[i][j] = np.sqrt(2 / n)*np.cos(np.pi*(2*j+1)*i/(2*n))
    dct = np.matmul(np.matmul(m,ar),np.transpose(m))
    M = m
    return dct
def BackDCT(dc):
    global M
    mat = M
    res = np.matmul(np.matmul(np.transpose(mat), dc), mat)
    res = np.int32(res)
    return res
lumQ = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]])


chromQ = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]])
def GetQuantMatrix(quality, isL = True):
    scaleFactor = 5000 / quality if quality <= 50 else 200 - quality * 2
    if isL:
        lQM = lumQ.copy()
        for i in range(8):
            for j in range(8):
                lQM[i][j] = np.ceil((lQM[i][j] * scaleFactor + 50)/100)
        return lQM
    else:
        cQM = chromQ.copy()
        for i in range(8):
            for j in range(8):
                cQM[i][j] = np.ceil((cQM[i][j] * scaleFactor + 50)/100)
        return cQM

# Квантование и обратное квантование
def Quantize(Cdct, isL = True, Q = 50):
    quant = GetQuantMatrix(Q, isL) if isL else GetQuantMatrix(50, False)
    for i in range(8):
        for j in range(8):
            Cdct[i][j] = np.round(Cdct[i][j] / quant[i][j])
    return np.int8(np.matrix.round(Cdct, 0))

def Dequantize(Cdct, isLuminance = True, Q = 50):
    quantMatrix = GetQuantMatrix(Q, isLuminance) if isLuminance else GetQuantMatrix(50, False)
    Cdct = np.float32(Cdct)
    for i in range(8):
        for j in range(8):
            Cdct[i][j] = np.round(Cdct[i][j] * quantMatrix[i][j])
    return Cdct


def Blocks(arr, n = 8, k = 8):
    new = np.array(arr)
    r, h = new.shape
    ret = []
    for i in range(0, r, n):
        subarray = np.array(new[i:i + n])
        # print(subarray)
        subarray = np.hsplit(subarray, h // k)
        # print(np.array(subarray))
        ret += subarray[::1]
    return np.array(ret)

def Merge(arr, n, k):
    new = np.array(arr)
    h, r, c = new.shape
    ret = []
    for i in range((n // r)):
        subarray = tuple(new[i * (k // r):(i + 1) * (k // r)])
        # print(subarray)
        ret.append(np.concatenate(subarray, axis=1))
        # print(ret)
    ret = np.concatenate(ret, axis=0)
    return ret
def GetResultArrays(__path, qual):
    # img = rgbtoycbcr(np.array(Image.open(__path)))
    # i, j, k = img.shape
    # if i % 16 != 0:
    #     img = np.pad(img, ((0, 16 - i % 16), (0, 0), (0, 0)), 'constant')
    # if j % 16 != 0:
    #     img = np.pad(img, ((0, 0), (0, 16 - j % 16), (0, 0)), 'constant')
    # img2 = np.uint8(downsamplingapprox(img))
    img = Image.open(__path)
    img = np.array(img)
    Y,Cb,Cr = rgbtoycbcr(img)
    Y = np.reshape(Y, (len(img), len(img[0])))
    Cb = np.reshape(Cb, (len(img), len(img[0])))
    Cr = np.reshape(Cr, (len(img), len(img[0])))
    Cb = np.int32(downsamplingapprox(Cb))
    Cr =  np.int32(downsamplingapprox(Cr))
    Cb = Cb-128
    Cr = Cr-128
    Y = np.int32(Y)-128
    Cb = Blocks(Cb)
    Cr = Blocks(Cr)
    Y = Blocks(Y)
    resY = []
    resCb = []
    resCr = []
    for i in range(len(Y)):
        Y[i] = DCT(Y[i])
        Y[i] = Quantize(Y[i], Q=qual)
        resY.append(diag(Y[i]))
    for i in range(len(Cb)):
        Cb[i] = DCT(Cb[i])
        Cb[i] = Quantize(Cb[i], False, Q=qual)
        resCb.append(diag(Cb[i]))
    for i in range(len(Cr)):
        Cr[i] = DCT(Cr[i])
        Cr[i] = Quantize(Cr[i], False, Q=qual)
        resCr.append(diag(Cr[i]))
    # print(len(img2), len(img2[0]))
    return resY, resCb, resCr, len(img)//2, len(img[0])//2

# вывод изображения из полученных массивов
def FromResultArrays(Yr, Cbr, Crr, hor, ver, qual):
    Y = [None for i in range(len(Yr))]
    Cb = [None for i in range(len(Cbr))]
    Cr = [None for i in range(len(Crr))]
    # print(resYc[0])
    # print(zigzag.inverseZigZag(resYc[0], 8))
    for i in range(len(Y)):
        Y[i] = matri(Yr[i], 8, 8)
        Y[i] = np.int32(Y[i])
        Y[i] = Dequantize(Y[i], Q=qual)
        Y[i] = BackDCT(Y[i])
        # Yc.append(a)
    for i in range(len(Cb)):
        Cb[i] = matri(Cbr[i], 8,8)
        Cb[i] = np.int32(Cb[i])
        Cb[i] = Dequantize(Cb[i], False, Q=qual)
        Cb[i] = BackDCT(Cb[i])
        # Cb.append(a)
    for i in range(len(Cr)):
        Cr[i] = matri(Crr[i], 8,8)
        Cr[i] = np.int32(Cr[i])
        Cr[i] = Dequantize(Cr[i], False, Q=qual)
        Cr[i] = BackDCT(Cr[i])
        # Cr.append(a)

    Y = np.array(Y)
    Cb = np.array(Cb)
    Cr = np.array(Cr)
    Yc = Merge(Y, ver*2, hor*2)
    Cb = Merge(Cb, ver, hor)
    Cr = Merge(Cr, ver, hor)

    img3 = []
    Cb = amsampling(Cb)
    Cr = amsampling(Cr)
    for i in range(len(Yc)):
        for j in range(len(Yc[0])):
            img3.append([Yc[i][j],Cb[i][j],Cr[i][j]])
    img3 = np.array(img3).reshape((len(Yc),len(Yc[0]),3))
    img3 = ycbcrtorgb(img3 + 128)
    img3 = np.array(img3)
    # img2 = np.concatenate((Yc, Cb, Cr), axis=3)
    img = Image.fromarray(img3.astype(np.uint8), mode="RGB")
    img.show()
    img.save('pic1.jpeg')

# кодирование и декодирование RLE
def run_length_encoding(s):
    encoded = []
    count = 1
    flag = chr(256)
    strlen = len(s)
    for i in range(1, strlen):
        if s[i] == s[i - 1]:
            count += 1
        else:
            if count == 1:
                encoded.append(s[i - 1])
            else:
                encoded.append(count)
                encoded.append(-1)
                encoded.append(s[i - 1])
            count = 1
    if count == 1:
        encoded.append(s[len(s) - 1])
    else:
        encoded.append(count)
        encoded.append(-1)
        encoded.append(s[len(s) - 1])
    res = ''
    for i in encoded:
        if (i>=0):
            res+=str(i)+' '
        else:
            res += flag
    return res


def run_length_decoding(string):
    decoded = ''
    flag = chr(256)
    i = 0
    strlen = len(string)
    while (i < (strlen-1)):
        if (i!=0)and(string[i] == flag):
            r = l = 0
            while(string[i+r]!=' ')and((i+r)<len(string)-1):
                r+=1
            while (string[i - 2 - l] != ' ')and(i - 1 - l>0):
                l+=1
            decoded = decoded[:(len(decoded)-1-l)]
            decoded += int(string[(i - 1 - l):(i - 1)]) * (string[(i + 1):(i + r)]+' ')
            i+=r+1
        else:
            decoded += string[i]
            i+=1
    return decoded

# сжатие изображения и запись его в файл
def Compress(__path, qual):
    Yr, Cbr, Crr, img2v, img2h = GetResultArrays(__path, qual)
    # print(resYc)
    # print(len(resYc), len(resYc[0]))
    print(len(Yr), len(Cbr), len(Crr))
    eY = []
    for i in Yr:
        eY += [int(j) + 128 for j in i]
    eY = run_length_encoding(eY)
    # eY = ''.join([chr(i) for i in eY])
    eCb = []
    for i in Cbr:
        eCb += [int(j) + 128 for j in i]
    eCb = run_length_encoding(eCb)

    eCr = []
    for i in Crr:
        eCr += [int(j) + 128 for j in i]
    eCr = run_length_encoding(eCr)
    print(len(eY), len(eCb), len(eCr))
    leny = len(eY).to_bytes(4, byteorder='big')
    lencb = len(eCb).to_bytes(4, byteorder='big')
    lencr = len(eCr).to_bytes(4, byteorder='big')
    quality = qual.to_bytes(1, byteorder='big')
    hor = img2h.to_bytes(2, byteorder='big')
    vert = img2v.to_bytes(2, byteorder='big')
    data = eY + eCb + eCr
    data = data.encode('utf-8')
    data = leny + lencb + lencr + quality + hor + vert + data
    with open('compressed.bin', 'wb') as f:
        f.write(data)
        f.close()

# распаковка изображения
def Show(__path):
    with open(__path, 'rb') as f:
        data = f.read()
        f.close()
    leny = int.from_bytes(data[0:4], byteorder='big')
    lencb = int.from_bytes(data[4:8], byteorder='big')
    lencr = int.from_bytes(data[8:12], byteorder='big')
    qual = int.from_bytes(data[12:13], byteorder='big')
    hor = int.from_bytes(data[13:15], byteorder='big')
    vert = int.from_bytes(data[15:17], byteorder='big')
    data = data[17:].decode('utf-8')
    eY = data[0:leny]
    eCb = data[leny:leny + lencb]
    eCr = data[leny + lencb:leny + lencb + lencr]
    print(leny, lencb, lencr)
    img2v, img2h = vert, hor
    eY = run_length_decoding(eY)
    eCb = run_length_decoding(eCb)
    eCr = run_length_decoding(eCr)
    res1 = list(map(int, eY.split()))
    res2 = list(map(int, eCb.split()))
    res3 = list(map(int, eCr.split()))
    resCb = []
    resYc = []
    resCr = []
    for i in res1:
        resYc += [i - 128]
    for i in res2:
        resCb += [i - 128]
    for i in res3:
        resCr += [i - 128]
    print(len(resYc), len(resCb), len(resCr))
    if len(resYc) % 64 != 0:
        resYc += [0] * (64 - len(resYc) % 64)
    resYc = [resYc[i:i + 64] for i in range(0, len(resYc), 64)]
    resCb = [resCb[i:i + 64] for i in range(0, len(resCb), 64)]
    resCr = [resCr[i:i + 64] for i in range(0, len(resCr), 64)]
    FromResultArrays(resYc, resCb, resCr, img2h, img2v, qual)

# Compress("nelena.jpg", 80)
# Show("compressed.bin")
# Compress("pink.jpg", 80)
# Show("compressed.bin")
Compress("rand.jpeg", 80)
Show("compressed.bin")

# img = img.resize((400,400))
# arr = imgtoraw(img)
# print(arr)
# y,Cb,Cr = rgbtoycbcr(arr)
# y1 = np.reshape(y, (len(arr), len(arr[0])))
# Cb1 = np.reshape(Cb, (len(arr), len(arr[0])))
# Cr1 = np.reshape(Cr, (len(arr), len(arr[0])))
# res = []
# for i in range(len(y)):
#     res.append([y[i],Cb[i],Cr[i]])
# res = np.array(res)
# res = np.reshape(res,(len(arr),len(arr[0]),3))
# immg = Image.fromarray(res.astype(np.uint8), mode='YCbCr')
# # img.show()
# immg.show()
# res2 = downsamplingdel(res)
# immg2 = Image.fromarray(res2.astype(np.uint8), mode='YCbCr')
# immg2.show()
# # res1 = downsamplingapprox(res)
# # immg1 = Image.fromarray(res1.astype(np.uint8), mode='YCbCr')
# # immg1.show()
# # immg.save("C:/Users/user/Desktop/lena1.jpg")
# res0 = amsampling(res2)
# immg0 = Image.fromarray(res0.astype(np.uint8), mode='YCbCr')
# immg0.show()
# immg.save("C:/Users/user/Desktop/lena1.jpg")
# img.close()
# immg.close()

#
# img = Image.open("C:/Users/user/Desktop/lena1.jpg")
# img = img.convert('YCbCr')
# #img = img.resize((400,200))
# arr = imgtoraw(img)
# R,G,B = ycbcrtorgb(arr)
# r = np.reshape(R, (len(arr), len(arr[0])))
# g = np.reshape(G, (len(arr), len(arr[0])))
# b = np.reshape(B, (len(arr), len(arr[0])))
# res = []
# for j in range(len(R)):
#     res.append([R[j],G[j],B[j]])
# res = np.array(res)
# print(len(r))
# res = np.reshape(res,(len(arr),len(arr[0]),3))
# immg = Image.fromarray(res.astype(np.uint8), mode='RGB')
# img.show()
# immg.show()
# #immg.save("C:/Users/user/Desktop/lena1.jpg")
# img.close()
# immg.close()

# ar = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
# print(ar)
# d = diag(ar)
# print(d)
# print(matri(d,len(ar),len(ar[0])))
