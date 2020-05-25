import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import filedialog as fd
from scipy.stats.stats import pearsonr
import os

####################################
############  Полное описание проекта
############  и последние обновления можно найти в репозитории разработчика:
############  https://github.com/maratdzhan/dependencies-F2-from-indices-T
####################################

COLUMNS = ['YEAR', 'HOUR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', \
           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
MONTHS_ARR = COLUMNS[2:]
HOUR_A = np.arange(0,24,1)
MAX_YEAR = 2018
OBS_YEAR = 2019
yticks_value = 0.25
indices_filename = "ind.txt";
frequences_filename = "freq.txt";
result_file = "table_result.txt"

emptyStringChar = 6;

data = pd.DataFrame();
data_tm = pd.DataFrame();


############################################################
##
def RU_MON(MON):
    RM = ["Январь", "Февраль", "Март", "Апрель","Май","Июнь","Июль",
         "Август","Сентябрь","Октябрь","Ноябрь","Декабрь"];
    return (RM[MONTHS_ARR.index(MON)]);

## 
def SaveTable(data, filename, nl = 0):
    file = open(filename, 'w');
    for line in data:
        if (nl):
            line = str(line) + '\n';
        else:
            line = str(line);
        file.write(line);
    file.close()
############################################################

##
def GetFrequencesFile():
    fn = fd.askopenfilename()
    print(fn)
    if not fn:
        fn = "freq.txt";
    else:
        Preproc(fn);
    frequences_filename = fn;
    print("Файл частот выбран");

def GetIndicesFile():
    fn = fd.askopenfilename()
    print(fn);
    if not fn:
        fn = "ind.txt";
    else:
        ParseIndices(fn);
    indices_filename = fn;
    print("Файл индексов выбран");
    
##
def ParseIndices(file):
    file_content = []
    file_obj = open(file, 'r')
    table = False;
    for line in file_obj:
        if (line.find("=") != -1): 
            if (table):
                table = False;
            else:
                table = True;
        else:
            if table:
                line = line.strip() + "\n";
                file_content.append(line);
    WriteIndices(file_content);
    
##
def WriteIndices(filec):
    for i in range(0,len(filec)):
        filec[i] = filec[i].replace('^', ' ').replace('*', ' ')
    SaveTable(filec, "ind.txt");
    indices_filename = "ind.txt";


def FindWords(line):
    mystr = line.replace(' ', '')
    mystr = mystr.replace('.', '')
    mystr = mystr.replace('\n', '')
    t = mystr.isdigit()
    return ((t));

##
def Preproc(filename):
    temp_data_array = [];
    year = 0
    temp_data = open(filename, 'r');
    for line in temp_data:
        if (FindWords(line)):
            if (len(line) < 6):
                year = line.replace('\n', '');
            else:
                line = line.strip() + "\n";
                line = year + " " + line
                temp_data_array.append(line);
    temp_data.close()
    SaveTable(temp_data_array, 'freq.txt');
    frequences_filename = "freq.txt";

def Import():
    data_tm = pd.read_csv(frequences_filename, sep = ' ', \
                     names = COLUMNS, skipinitialspace = True, index_col = 'YEAR')
    fcd = ['YEAR', 'JAN', 'FEB','MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    data = pd.read_csv(indices_filename, sep = ' ', names = fcd, skipinitialspace = True,  index_col = 'YEAR')
    return data, data_tm;
    
def DropEmpty(data):
    for i in data.columns:
        if (i not in COLUMNS):
            data.drop(i, axis = 1, inplace = True)
    return data;

def cube_funct(coefficients, arguments):
    f = (coefficients[0]+coefficients[1]*arguments+\
        coefficients[2]*(arguments**2)+coefficients[3]*(arguments**3))
    return f;


def Plotting(t, f2, x, y,  ccs, correlation_coeff, points, MONTHS, HOURS, path, isShow = False, label1 = 'f2(t)', label2 = 'cubic regression'):
    plt.figure(figsize=(15,10))
    plt.plot(t, f2,'g^',label=label1, alpha = 0.75)
    plt.plot(x, y, 'r.', label=label2, alpha = 0.75)
    plt.plot(points[0, 0],points[0, 1], 'bD',\
             label = '2019 prediction ({};{})'.format(points[0,0], points[0,1]));
    plt.plot(points[1, 0],points[1, 1], color = 'magenta', marker ='P',\
             label = "2019 factual ({};{})".format(points[1,0], points[1, 1]));
    plt.grid(True, color = 'blue',alpha = 0.15)
    plt.annotate("foF2$_m$$_e$$_d$ = ({:.3e})$T^3$ + ({:.3e})$T^2$ + ({:.3e})$T$ + {:.4f}, \n\n\
    Pearson's correlation coefficient: {:.4f}".format(ccs[3],ccs[2],ccs[1],ccs[0], correlation_coeff),\
             xy=(0, 1), xytext=(12, -12), va='top',\
             xycoords='axes fraction', textcoords='offset points', fontsize = 16)
    plt.title("{}, {} o'clock".format(MONTHS, HOURS), fontsize= 20)
    plt.xlabel("T", fontsize = 16)
    plt.ylabel("F2", fontsize = 16)
    plt.legend(loc = 'lower right', fontsize = 16)
    if not (os.path.exists(path)):
        os.mkdir(path)
    name = path + '/' + MONTHS + '-' + str(HOURS) + '.png'
    plt.savefig(name)
    if isShow:
        plt.show()
    plt.close()



def mainCycle(data, data_tm, path = 'graphs'):
    out_table = [];
    ############################################################
    out_table.append("Month,Hour,a,b,c,d,Pearson R,dF\n")
    dF2 = [];
    ############################################################
    for MON in data.columns:
        for HRS in HOUR_A:
            print("{}-{}".format(MON, HRS));
            dF = data_tm.query('HOUR == @HRS & YEAR <= @MAX_YEAR & {} > 0'.format(MON))[MON]
            dT = data.loc[data.index.isin(dF.index)][MON]
            t = dT.values[:]
            f = dF.values[:]
            t1 = np.array(t);
            f1 = np.array(f);
            z = zip(t,f)
            z1 = zip(t1,f1);
            f.sort()
            t.sort()
            zs = sorted(z, key = lambda tup: tup[0])
            zs1 = sorted(z1, key = lambda tup: tup[0])
            f = [z[1] for z in zs]
            t = [z[0] for z in zs]
            f1 = [z1[1] for z1 in zs1]
            t1 = [z1[0] for z1 in zs1]
            t1 = np.array(t1)
            t = np.array(t)
            lv = np.arange(-30, t[len(t1) - 1], 1)
            tt = np.polyfit(t, f, 3)
            tt = list(reversed(tt))
            cube_polyfit = cube_funct(tt,lv)
            corr_coeff = pearsonr(t1, f1)[0]
            accumulated_info = "{},{},{:.5e},{:.5e},{:.5e},{:.5f},{:.5f}\
            \n".format(MON, HRS, tt[3], tt[2], tt[1], tt[0], corr_coeff)
            ############################################################
            ## PREDICT
            dFP = data_tm.query('HOUR == @HRS & YEAR == @OBS_YEAR & {} != "NaN"'.format(MON))[MON]
            dTP  = data.loc[data.index.isin(dFP.index)][MON]
            kp = dTP.values[:]
            resp = cube_funct(tt, kp)
            predict = np.array([kp,resp])
            ## FACT
            dFF = data_tm.query('HOUR == @HRS & YEAR == @OBS_YEAR & {} != "NaN"'.format(MON))[MON]
            dTF  = data.loc[data.index.isin(dFP.index)][MON]
            fact = np.array([dTF.values[:],dFF.values[:]])
            points = np.array([predict,fact])
            devF = fact[1,0] - predict[1,0];
            dF2.append(devF);
            accumulated_info = "{},{},{:.5e},{:.5e},{:.5e},{:.5f},{:.5f},{:.5f}\
            \n".format(MON, HRS, tt[3], tt[2], tt[1], tt[0], corr_coeff, devF)
            out_table.append(accumulated_info)
            ############################################################
            Plotting(t1, f1, lv, cube_polyfit, tt, corr_coeff, points, MON, HRS, path, False)
    ############################################################
    mdF = np.mean(dF2);
    ddF = np.var(dF2);
    s = "mean dF value: {}\n var dF value: {}".format(mdF, ddF);
    md_data = np.array([s])
    SaveTable(md_data, "statistic.txt",1);
    SaveTable(dF2, "dF2_values.txt",1);
    ############################################################
    return out_table;


def SetDependencies():
    data, data_tm = Import();
    data = DropEmpty(data);
    data_tm = DropEmpty(data_tm);
    ot = mainCycle(data, data_tm);
    SaveTable(ot,"result_table.txt");


def GetDaily(T_val, path_dir = "daily_graph"):
    rt = pd.read_csv(result_file);
    F2 = []
    for MON in MONTHS_ARR:
        mon_tab = rt.query('Month == @MON');
        for HR in HOUR_A:
            koef = list(reversed(mon_tab.iloc[HR,2:6].to_list()))
            F2.append(cube_funct(koef, T_val));
        plt.figure(figsize = (14,10));
        plt.plot(HOUR_A, F2, 'b-o');
        plt.title("Дневной ход медианной частоты F2 для месяца {} при индексе T = {}"\
                  .format(RU_MON(MON), T_val));
        plt.grid(color = 'blue', alpha = 0.2);
        plt.xlabel("Время суток, ч.", fontsize = 16)
        plt.ylabel("Медианные частоты F2, МГц", fontsize = 16)
        plt.xticks(HOUR_A)
        plt.yticks(np.arange(min(F2), max(F2) + yticks_value, yticks_value))
        if not (os.path.exists(path_dir)):
            os.mkdir(path_dir)
        path = path_dir + '/' + str(MONTHS_ARR.index(MON)) + '_' + MON + '_' + str(T_val) + '.png'
        plt.savefig(path);
        plt.close();
        F2.clear();
        

w = 200;
h = 200;
root = tk.Tk()
root.title("F2P")
root.geometry('300x150+{}+{}'.format(w, h))
f = tk.Frame()
bi = tk.Button(text="Load indices", command=GetIndicesFile);
bf = tk.Button(text="Load freq", command=GetFrequencesFile);
br = tk.Button(text="Run", command = SetDependencies);

bi.pack(padx = 10, pady = 10);
bf.pack(padx = 10, pady = 10);
br.pack(padx = 10, pady = 10)


root.mainloop()