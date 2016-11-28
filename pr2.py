#Project 2 Stereo Analysis
#Matt Schmittle

import cv2
import copy as cp
import numpy as np
import numpy.ma as ma
import math
np.set_printoptions(threshold='nan')

def sad(w1,w2):
    r,c = w1.shape
    calc = np.zeros((r,c))
    w1m = w1.mean()
    w2m = w2.mean()
    for row in range(0,r):
        for col in range(0,c):
            calc[row,col] = abs((w1[row,col]-w1m) - (w2[row,col]-w2m))
    return calc.sum()

def ssd(w1,w2):
    r,c = w1.shape
    calc = np.zeros((r,c))
    w1m = w1.mean()
    w2m = w2.mean()
    for row in range(0,r):
        for col in range(0,c):
            calc[row,col] = ((w1[row,col]-w1m) - (w2[row,col]-w2m))**2
    return calc.sum()

def ncc(w1,w2):
    r,c = w1.shape
    n = r*c
    w1m = w1.mean()
    w2m = w2.mean()
    if w1.std()==0:
        w1[0,0] = w1[0,0]+1
    if w2.std()==0:
        w2[0,0] = w2[0,0]+1
    calc1 = np.zeros((r,c))
    calc2 = np.zeros((r,c))
    calc3 = np.zeros((r,c))
    for row in range(0,r):
        for col in range(0,c):
            calc1[row,col] = ((w1[row,col]-w1m)*(w2[row,col]-w2m))
            calc2[row,col] = (w1[row,col]-w1m)**2
            calc3[row,col] = (w2[row,col]-w2m)**2
    bottom = math.sqrt(calc2.sum()*calc3.sum())
    return calc1.sum()/bottom    

def count_nonzeros(window):
    r,c = window.shape
    count = 0
    count3 = 0
    for row in range(0,r):
        for col in range(0,c):
            if window[row,col] >=20 and window[row,col] != 300:
                count+=1
            if window[row,col] == 300:
                count3+=1
    return count,count3

def averaging(window):
    r,c = window.shape
    sumw = 0
    count,other = count_nonzeros(window)
    for row in range(0,r):
        for col in range(0,c):
            if window[row,col] != 300 and window[row,col] > 20:
                sumw+= window[row,col]
    try:
        avg = int(sumw/count)
    except:
        avg = 20
    if avg <20:
        avg = 20
    return avg 
            
def interpolate(din):
    d = cp.copy(din)
    r,c = d.shape
    for row in range(0,r):
        for col in range(0,c):
            if d[row,col] < 20:
                correct_window = False
                size = 3
                while not(correct_window):
                    window = make_window(size,d,row,col)
                    nonz,count3 = count_nonzeros(window)
                    rw,cw = window.shape
                    if nonz<int(((size**2)-count3)/2)and rw<r/8 and cw<c/8:
                        size += 2
                    else:
                        correct_window = True
                d[row,col] = averaging(window)
    return d

def nearest(d,row,col):
    r,c = d.shape
    search = 1
    res = 1
    found = False
    avg = 0
    count = 0
    while not(found):
        try:
            if d[row,col-(res*search)] != 0 and d[row,col-(res*search)] != 300 :
                avg += d[row,col-(res*search)] 
                count+=1
        except:
            avg+=0
        try:
            if d[row,col+(res*search)] != 0 and d[row,col+(res*search)] != 300:
                avg += d[row,col+(res*search)] 
                count+=1
        except:
            avg+=0
        try:
            if d[row-(res*search),col] != 0 and d[row-(res*search),col] != 300:
                avg += d[row-(res*search),col] 
                count+=1
        except:
            avg+=0
        try:
            if d[row+(res*search),col] != 0 and d[row+(res*search),col] != 300:
                avg += d[row+(res*search),col] 
                count+=1
        except:
            avg+=0
        try:
            if d[row+(res*search),col+(res*search)] != 0 and d[row+(res*search),col+(res*search)] != 300:
                avg += d[row+(res*search),col+(res*search)] 
                count+=1
        except:
            avg+=0
        try:
            if d[row-(res*search),col-(res*search)] != 0 and d[row-(res*search),col-(res*search)] != 300:
                avg += d[row-(res*search),col-(res*search)] 
                count+=1
        except:
            avg+=0
        try:
            if d[row+(res*search),col-(res*search)] != 0 and d[row+(res*search),col-(res*search)] != 300:
                avg += d[row+(res*search),col-(res*search)] 
                count+=1
        except:
            avg+=0
        try:
            if d[row-(res*search),col+(res*search)] != 0 and d[row-(res*search),col+(res*search)] != 300:
                avg += d[row-(res*search),col+(res*search)] 
                count+=1
        except:
            avg+=0

        if avg != 0:
            found = True
            avg = int(avg/count)
            if avg <20:
                avg = 20
        else:
            if search < r/8:
                search+=1
            else:
                found = True
                avg = 20
    return avg

def interpolate_feature2(din,g):
    d = cp.copy(din)
    r,c = d.shape
    for thresh in range(60,150,10):
        t,tg = cv2.threshold(g,thresh,255,cv2.THRESH_TOZERO)
        for row in range(0,r):
            for col in range(0,c):
                if tg[row,col] ==0 and d[row,col] ==0:
                    near = nearest(d,row,col)
                    replacement = np.full((3,3),near,np.uint64)                
                    try:
                        d[row-1:row+2,col-1:col+2] =replacement 
                    except:
                        rw,dw = d[row-1:row+2,col-1:col+2].shape 
                        replacement = np.full((rw,dw),near,np.uint64)                
                        d[row-1:row+2,col-1:col+2] =replacement
    for row in range(0,r):
        for col in range(0,c):
            if d[row,col] == 0:
                d[row,col] = nearest(d,row,col)
    dum,maxd,mind = norm(d,20,255,2,'feature')

    return d,maxd,mind
                
        
            

def bilinear(window,row,col):
    nonzeros = np.nonzero(window)
    closest = []
    for nz in range(0,len(nonzeros[0])):
        rdist = abs(row - nonzeros[0][nz]) 
        cdist = abs(col - nonzeros[1][nz]) 
        dist = math.sqrt(rdist**2 + cdist**2)
        if len(closest) < 5:
            closest.append([nonzeros[0][nz],nonzeros[1][nz],dist])
        else:
            done = False
            for x in closest:
                if x[2] > dist and not(done):
                    x = [nonzeros[0][nz],nonzeros[1][nz],dist]
                    done = True
        cols = [closest[0][1],closest[1][1],closest[2][1],closest[3][1]]
        vals = [closest[0][0],closest[1][0],closest[2][0],closest[3][0]]
        row = [closest[0][2],closest[1][2],closest[2][2],closest[3][2]]

        array = []
        twos = []
             

def interpolate_bilinear(din):
    d = cp.copy(din)
    r,c = d.shape
    for row in range(0,r):
        for col in range(0,c):
            if d[row,col] <20:
                correct_window = False
                size = 3
                while not(correct_window):
                    window = make_window(size,d,row,col)
                    count = len(np.nonzero(window)[0])
                    if count >4:
                        correct_window = True
                d[row,col] = bilinear(window,row,col)
    return d
            
def compare(a,b,method):
    if method != 'ncc':
        return a < b
    else:
        a_score = abs(1-a)
        b_score = abs(1-b)
        return a_score < b_score

def map_val(x,mini,maxi,mino,maxo):
    outrange = maxo-mino
    drange = maxi-mini
    return ((x-mini)*outrange)/(drange+mino)

def remove_outliers(d,num,mino):
    din = cp.copy(d)
    m = din.mean()
    s = num*(din.std())
    r,c = din.shape
    for row in range(0,r):
        for col in range(0,c):
            if din[row,col] > m+s or din[row,col]<m-s:
                din[row,col] = mino 
    return din

def norm(din,mino,maxo,mul,technique):
    d = remove_outliers(din,mul,mino)
    maxi = d.max()
    mini = d.min()
    dout = cp.copy(d)
    r,c = dout.shape
    for row in range(0,r):
        for col in range(0,c):
            val = map_val(dout[row,col],mini,maxi,mino,maxo)
            if technique !='feature'and val<20:
                val = 20
            dout[row,col] = val
    return dout,maxi,mini

def multi_region_window(gc,size,method,indisp,maxd,mind,technique):
    disp,mini,maxi = norm(indisp,mind,maxd,2,technique)
    r,c = gc.shape
    num = c/18
    disparities = np.zeros((r,c/2))
    dr,dc = disparities.shape
    half_size = int(size/2)
    backwards = []
    for row in range(half_size,r,size):
        for col in range(half_size,c/2,size):
            window = make_window(size,gc,row,col)
            out_window = window
            window2 = make_window(size+2,gc,row,col)
            if method != "ncc":
                temp = [1000000,0,10000000]
            else:
                temp =[100000,-100000000,100000]
            if disp[row-1,col] != 0:
                val = disp[row-1,col]
            if disp[row-1,col-1] != 0:
                val = disp[row-1,col-1]
            if disp[row,col-1] != 0:
                val = disp[row,col-1]
            try:
                shift = (col-val)+c/2
            except:
                shift = col+(c/2)
            for colw in range(shift - num,shift):
                score = 100000
                temp_window = make_window(size,gc,row,colw)
                if method == "sad":
                   score = sad(window,temp_window)
                if method == "ssd":
                   score = ssd(window,temp_window)
                if method == "ncc":
                   score = ncc(window,temp_window)
                if compare(score,temp[0],method):
                    t2w = make_window(size+2,gc,row,colw)
                    if method == "sad":
                       score2 = sad(window2,t2w)
                    if method == "ssd":
                       score2 = ssd(window2,t2w)  
                    if method == "ncc":
                       score2 = ncc(window2,t2w)
                    if compare(score2,temp[2],method):
                        temp = [score,colw,score2]
            backwards.append([row,col,temp[1]])
            replacement = np.full((size,size),abs(col-(temp[1]-(c/2))),np.uint64)
            try:
                disparities[row-half_size:row+half_size+1,col-half_size:col+half_size+1] = replacement
            except:
                rr,rc = disparities[row-half_size:row+half_size+1,col-half_size:col+half_size+1].shape
                replacement = np.full((rr,rc),(col-(temp[1]-c/2)),np.uint64)
                disparities[row-half_size:row+half_size+1,col-half_size:col+half_size+1] = replacement
                
    disparities,maxdo,mindo = norm(disparities,20,255,2,technique)
    disparities = disparities.astype(np.uint8)
    return disparities,backwards,maxdo,mindo
                
def region_window(gc,size,method,technique):
    r,c = gc.shape
    num =c/18
    disparities = np.zeros((r,c/2))
    half_size = int(size/2)
    backwards = []
    count = 0
    for row in range(half_size,r,size):
        for col in range(half_size,c/2,size):
            window = make_window(size,gc,row,col)
            out_window = window
            window2 = make_window(size+2,gc,row,col)
            if method !='ncc':
                temp = [1000000,0,10000000]
            else:
                temp = [1000000,-10000000000,10000000]
            for colw in range((col+(c/2))-num,col+(c/2)): 
                score = 100000 
                temp_window = make_window(size,gc,row,colw) 
                if method == "sad":
                   score = sad(window,temp_window)
                if method == "ssd":
                   score = ssd(window,temp_window)
                if method == "ncc":
                   score = ncc(window,temp_window)
                if compare(score,temp[0],method):
                    t2w = make_window(size+2,gc,row,colw)
                    if method == "sad":
                       score2 = sad(window2,t2w)
                    if method == "ssd":
                       score2 = ssd(window2,t2w)  
                    if method == "ncc":
                       score2 = ncc(window2,t2w)
                    if compare(score2,temp[2],method):
                        temp = [score,colw,score2]
            backwards.append([row,col,temp[1]])
            replacement = np.full((size,size),abs(col-(temp[1]-c/2)),np.uint64)
            try:
                disparities[row-half_size:row+half_size+1,col-half_size:col+half_size+1] = replacement
            except:
                rr,rc = disparities[row-half_size:row+half_size+1,col-half_size:col+half_size+1].shape
                replacement = np.full((rr,rc),abs(col-(temp[1]-c/2)),np.uint64)
                disparities[row-half_size:row+half_size+1,col-half_size:col+half_size+1] = replacement
               
                
    disparities,maxd,mind = norm(disparities,20,255,2,technique)
    disparities = disparities.astype(np.uint8)
    return disparities,backwards,maxd,mind

def backwards_check(gc,backwards,dmap,size,method): 
    #back = row,col,newcol
    d = cp.copy(dmap)
    r,c = gc.shape
    num = c/15
    replacement = np.full((size,size),0,dtype=np.uint8)
    half_size = int(size/2)
    if method == "descriptor":
        return back_descriptor(gc,backwards,d)
    else:
        for back in backwards:
            if method != "ncc":
                temp = [1000000,0]
            else:
                temp = [-1000000,0]
            window = make_window(size,gc,back[0],back[2])
            for col in range((back[2]-(c/2)),(back[2]-(c/2))+num):            
                    score = 1000000
                    temp_window = make_window(size,gc,back[0],col)
                    if method == "sad":
                       score = sad(window,temp_window)
                    if method == "ssd":
                       score = ssd(window,temp_window)
                    if method == "ncc":
                       score = ncc(window,temp_window)
                    if compare(score,temp[0],method):
                        temp = [score,col]

            if abs(temp[1]-back[1])>size:
                try:
                    d[back[0]-half_size:back[0]+half_size+1,back[1]-half_size:back[1]+half_size+1] = replacement 
                except:
                    rr,rc = d[back[0]-half_size:back[0]+half_size+1,back[1]-half_size:back[1]+half_size+1].shape
                    replacement = np.full((rr,rc),0,dtype=np.uint8)
                    d[back[0]-half_size:back[0]+half_size+1,back[1]-half_size:back[1]+half_size+1] = replacement 
                    

        return d 
            

def feature(gc,method,size,level):
    hc = cv2.cornerHarris(gc,3,3,.04)
    #hcn,maxh,minh = norm(hc,0,255,2)
    hcn = cv2.normalize(hc,0,255)
    x,y = hcn.shape
    lfeat = []
    for i in range(0,x):
        for j in range(0,y):
            if hcn[i][j] >0.5:
                if j < y/2:
                    lfeat.append([hcn[i][j],j,i])
    if method == "descriptor":
        return descriptor(lfeat,hcn)
    else:
        return window_based(lfeat,hcn,method,size,gc,level)

def window_based(lfeat,hcn,method,size,gc,level):
    r,c = hcn.shape
    num = c/18
    disparities = np.zeros((r,c/2))
    backwards = []
    for feat in lfeat:
        if method != "ncc":
            temp = [10000000,0,10000000]   
        else:
            temp = [-10000000,0,-10000000]
        window = make_window(size,gc,feat[2],feat[1])
        window2 = make_window(size+2,gc,feat[2],feat[1])
        try:
            first = (feat[1] - level[feat[2],feat[1]]) + (c/2) -num
            if first < c/2:
                first = c/2
            second = (feat[1] - level[feat[2],feat[1]]) + c/2 
        except:
            first = feat[1]+(c/2)-num
            if first < c/2:
                first = c/2
            second = feat[1]+(c/2)
        for col in range(first,second):
            score = 100000
            temp_window = make_window(size,gc,feat[2],col)
            if method == "sad":
               score = sad(window,temp_window)
            if method == "ssd":
               score = ssd(window,temp_window)
            if method == "ncc":
               score = ncc(window,temp_window)
            if compare(score,temp[0],method):
                t2w = make_window(size+2,gc,feat[2],col)
                if method == "sad":
                   score2 = sad(window2,t2w)
                if method == "ssd":
                   score2 = ssd(window2,t2w)
                if method == "ncc":
                   score2 = ncc(window2,t2w)
                if compare(score2,temp[2],method):
                    temp = [score,col,score2]
        disparities[feat[2],feat[1]] = abs(feat[1]-(temp[1]-c/2))
        backwards.append([feat[2],feat[1],temp[1]])
    disparities,maxd,mind = norm(disparities,20,255,11,'feature')
    disparities = disparities.astype(np.uint8)
    return disparities,backwards
    
def make_window(size,gc,row,col):
    half_size = int(size/2)
    window = np.zeros((size,size))
    for roww in range(0,size):
        for colw in range(0,size):
            offset_row = roww - half_size
            offset_col = colw - half_size
            try:
                if row+offset_row>=0 and col+offset_col>=0:
                    window[roww,colw] = gc[row+offset_row,col+offset_col]
                else:
                    window[roww,colw] = 300
            except:
                window[roww,colw] = 300
    return window

def expand(imagei,scale):
    image = np.multiply(imagei,2)
    r,c = image.shape
    out_image = np.zeros((r*scale,c*scale))
    for row in range(0,r):
        for col in range(0,c):
            out_image[row*scale,col*scale] = image[row,col]
            out_image[(row*scale)+scale/2,(col*scale) +scale/2] = image[row,col]
            out_image[(row*scale),(col*scale) +scale/2] = image[row,col]
            out_image[(row*scale)+scale/2,(col*scale)] = image[row,col]
    return out_image.astype(np.uint8)

def reduce(image,scale):
    r,c = image.shape
    out_image = np.zeros((r/scale,c/scale))
    for row in range(0,r/scale):
        for col in range(0,c/scale):
            out_image[row,col] = image[row*scale,col*scale]
    return out_image.astype(np.uint8)

def merge(weak,strong):
    r,c = strong.shape
    rw,cw = weak.shape
    for row in range(0,rw):
        for col in range(0,cw):
            if strong[row,col] < 20:
                strong[row,col] = weak[row,col]
    return strong
        
def multi_res(gc,g1,depth,size):
    images = []
    edisp = 0
    for num in range(0,depth):
        images.append(reduce(gc,2**num))
    for image in range(depth-1,-1,-1):
        matching = str(raw_input('What technique would you like to use (region,feature):\n'))
        method = raw_input('Matching method (sad,ssd,ncc,descriptor[feature-based-only]\n')
        print matching
        print method 
        print image
        if matching == "region":
            if image+1 == depth:
                disp,backwards,maxd,mind = region_window(images[image],size,method,matching)
                disp1 = backwards_check(images[image],backwards,disp,size,method)
                disp2 = interpolate(disp1)
                edisp = expand(disp2,2)
            else:
                disp,backwards,maxd,mind = multi_region_window(images[image],size,method,edisp,maxd,mind,matching)
                disp1 = backwards_check(images[image],backwards,disp,size,method)
                edisp  = interpolate(disp1)
                if image !=0:
                    edisp = expand(edisp,2)
        if matching == "feature":
            if image+1==depth:
                disp,backwards = feature(images[image],method,size,"outer")
                disp_back = backwards_check(images[image],backwards,disp,size,method)
                interp,maxd,mind = interpolate_feature2(disp_back,g1)
                edisp = expand(interp,2)
            else:
                disp,backwards = feature(images[image],method,size,edisp)
                disp_back = backwards_check(images[image],backwards,disp,size,method)
                edisp,maxd,mind = interpolate_feature2(disp_back,g1)
                if image !=0:
                    edisp = expand(edisp,2)
    return edisp
        
def back_descriptor(gc,lfeat,disp):
    hc = cv2.cornerHarris(gc,3,3,.04)
    hcn = cv2.normalize(hc,0,255)
    x,y = hcn.shape
    temp = []
    backwards = []
    for feat in lfeat:
        top = [0,1]
        for k in range(0,y/2):
            if abs(hcn[feat[0]][k] - hcn[feat[0]][feat[2]]) < top[1]:
                top =[k,abs(hcn[feat[0]][k] - hcn[feat[0]][feat[2]])]
        if abs(top[0]-feat[1])<5:
            disp[feat[0],feat[1]] = 0
    return disp 
    

def descriptor(lfeat,hcn):
    x,y = hcn.shape
    disparities = np.zeros((x,y/2))
    temp = []
    backwards = []
    for feat in lfeat:
        top = [0,1]
        second = feat[1]+(y/2)+50        
        if y<= feat[1]+(y/2)+50:
            second = y
        first = feat[1]+(y/2)-50        
        if y/2>= feat[1]+(y/2)-50:
            first = y/2
        for k in range(first,second):
            if abs(hcn[feat[2],k] - feat[0])<top[1]:
                top = [k,abs(hcn[feat[2],k] - feat[0])]
        disparities[feat[2],feat[1]] = abs(feat[1]-(top[0]-(y/2)))
        backwards.append([feat[2],feat[1],top[0]])

    disparities,maxd,mind=norm(disparities,20,255,2,'feature')
    disparities = disparities.astype(np.uint8)
    return disparities,backwards 

#Setup
img1 = cv2.imread('left.png')
img2 = cv2.imread('right.png')
g1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
c = np.concatenate((img1,img2),axis = 1)
gc = np.concatenate((g1,g2),axis = 1)
test =False 
if not(test):
    try:
        i = input('Welcome to PR2 Matt Schmittle.\nNote, this program is not optimized so computation times may vary. \nPRESS ENTER TO CONTINUE')
    except:
        multi = int(raw_input('Multi-resolution stereo analysis (1/0):\n'))
        if multi:
            level = int(raw_input('Number of levels:\n'))
        else:
            technique = raw_input('What technique would you like to use (region,feature):\n')
            method = raw_input('Matching method (sad,ssd,ncc,descriptor[feature-based-only]\n')
        size = int(raw_input('Template matching size (odd numbers only)\n'))

    print "Doing:"

    if multi:
        print "Multi Resolution " 
        result = multi_res(gc,g1,level,size)
    else:
        print technique+" "+method
        if technique == "region":
            disp,backwards,maxd,mind = region_window(gc,size,method,'region')
            disp1 = backwards_check(gc,backwards,disp,size,method)
            result = interpolate(disp1)
        if technique == "feature":
            disp,backwards = feature(gc,method,size,"outer")
            disp_back = backwards_check(gc,backwards,disp,size,method)
            cv2.imshow('back',disp_back)
            result,maxf,minf= interpolate_feature2(disp_back,g1)

    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Testing for failures

#feature Based testing
#string = "hi"
#multi_feature = multi_res(gc,g1,3,5,"ncc","feature")
#cv2.imshow('sigh',multi_feature)
if test:
    try:
        string = 'feat-sad-5'
        print("attempting " + string + "...")
        multi_feature = multi_res(gc,g1,3,5,"sad","feature")
        cv2.imshow(string,multi_feature)
    except:
        print(string + " failed")
    try:
        string = 'feat-ssd-5'
        print("attempting " + string + "...")
        multi_feature = multi_res(gc,g1,3,5,"ssd","feature")
        cv2.imshow(string,multi_feature)
    except:
        print(string + " failed")

    try:
        string = 'feat-ncc-5'
        print("attempting " + string + "...")
        multi_feature = multi_res(gc,g1,3,5,"ncc","feature")
        cv2.imshow(string,multi_feature)
    except:
        print(string + " failed")
    try:
        string = 'feat-descriptor-5'
        print("attempting " + string + "...")
        multi_feature = multi_res(gc,g1,3,5,"descriptor","feature")
        cv2.imshow(string,multi_feature)
    except:
        print(string + " failed")

    #region based testing 
    try:
        string = 'region-sad-5'
        print("attempting " + string + "...")
        multi_region = multi_res(gc,g1,3,5,"sad","region")
        cv2.imshow(string,multi_region)
    except:
        print(string + " failed")

    try:
        string = 'region-ssd-5'
        print("attempting " + string + "...")
        multi_region = multi_res(gc,g1,3,5,"ssd","region")
        cv2.imshow(string,multi_region)
    except:
        print(string + " failed")

    try:
        string = 'region-ncc-5'
        print("attempting " + string + "...")
        multi_region = multi_res(gc,g1,3,5,"ncc","region")
        cv2.imshow(string,multi_region)
    except:
        print(string + " failed")

cv2.waitKey(0)

