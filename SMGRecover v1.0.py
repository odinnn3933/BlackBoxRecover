import pandas as pd
import numpy as np
from collections import defaultdict
import graphviz
from itertools import combinations
import math
from PysRProcess import symbolic_regression
from Simulate import eqprocess_with_namelist

global IntervalTime ,accuracylow

accuracylow = defaultdict(list)

class DataExcelFrames:
    def __init__(self,name,datatype,data):
        self.name = name
        self.dtype = datatype
        self.body = data

class Project_Data:
    def __init__(self,time):
        self.time = time
        self.input = defaultdict(list)
        self.output = defaultdict(list)
    ## 这里字典的处理需要重新查询
    def add_input(self,Name,Input_Element):
        self.input[Name].append(Input_Element)
        
    def add_output(self,Name,Output_Element):
        self.output[Name].append(Output_Element)

    def Name_List (self):
        NameList = []
        for key in self.input.keys():
            NameList.append(key)
        for key in self.output.keys():
            NameList.append(key)
        return NameList
    
class state_def: ## 这里需要根据变量名来定义状态
    def __init__(self, name):
        self.name = name
        self.state = []
        self.transition = []

    def add_state(self, state):
        if state not in self.state:
            self.state.append(state)

    def add_transition(self,Transition):
        self.transition.append(Transition)
    
    def SameTransition(self):
        ProcessIndex = []
        TransitionIDex = []
        for i in range(len(self.transition)):
            if i in ProcessIndex:
                continue
            ProcessIndex.append(i)
            TransitionIDex.append([i])
            for j in range(i+1,len(self.transition)):
                if j in ProcessIndex:
                    continue
                if self.transition[i].SourceState == self.transition[j].SourceState and self.transition[i].TargetState == self.transition[j].TargetState:
                    ProcessIndex.append(j)
                    TransitionIDex[-1].append(j)
        return TransitionIDex

class Trans_def:
    def __init__(self, StateS,StateT,Index):
        self.SourceState = StateS
        self.TargetState = StateT
        self.Condition = []
        self.Identify = defaultdict(list)
        self.Timer = defaultdict(list)
        self.ConFlag = False
        self.TimerFlag = False
        self.HappenTime = [Index]
        self.StayCondition = []
        self.Counter = []
    
    def add_condition(self, condition):
        self.Condition.append(condition)
        self.ConFlag = True

    def add_timer(self, key,Time,Value):
        self.Timer[key].append([Time,Value])
        self.TimerFlag = True

    def add_identify(self, key, value):
        self.Identify[key].append(value)
    
    def add_happen_time(self, time):
        self.HappenTime.append(time)

def group_consecutive_values(lst): ## 将连续的相同值分组
    if not lst:
        return []

    grouped_list = []
    current_group = [lst[0]]

    for value in lst[1:]:
        if value == current_group[-1]:
            current_group.append(value)
        else:
            grouped_list.append(current_group)
            current_group = [value]

    grouped_list.append(current_group)
    return grouped_list

def format_timestamp_to_seconds(timestamp):
    total_seconds = timestamp.total_seconds()
    return f"{total_seconds:.1f} s"

def ReadExcelFiles(Path):
    data_frames_project = []
    
    data_frames = []
    df = pd.read_excel(Path)
        
    for Col in df.columns:
        Data_insert = DataExcelFrames(Col,df[Col].dtype,[data for data in df[Col]])
        data_frames.append(Data_insert)
    data_frames_project.append(data_frames)
    return data_frames_project

def FindVariety(Data,Num):
    index = 0
    threshold = 0.0001
    Data = Data[0]  ## 只取第一个DataFrame
    OPData = Project_Data(Data[0].body)
    for data in Data[1:]:
        index += 1
        if index < Num:
            AProcess = group_consecutive_values(data.body)
            OPData.add_input(data.name,AProcess)
        else:
            if data.dtype != 'bool':
                eq,loss,indexlist = symbolic_regression(data.name,Data)  ## 这里需要进行符号回归
                if loss < threshold:
                    eq1 = data.name + "=" + data.name
                    grouped_list = []
                    if 0 in indexlist:
                        current_group = [eq]
                    else:
                        current_group = [eq1]
                    for i in range(1,len(data.body)):
                        if i in indexlist and current_group[-1] == eq:
                            current_group.append(eq)
                        elif i in indexlist and current_group[-1] != eq:
                            grouped_list.append(current_group)
                            current_group = [eq]
                        if i not in indexlist and current_group[-1] == eq1:
                            current_group.append(eq1)
                        elif i not in indexlist and current_group[-1] != eq1:
                            grouped_list.append(current_group)
                            current_group = [eq1]
                    AProcess = grouped_list
                else:
                    AProcess = group_consecutive_values(data.body)
            else:
                AProcess = group_consecutive_values(data.body)
            OPData.add_output(data.name,AProcess)
    return OPData

def OutputProcess(time,output):
    StateTimeDic = defaultdict(list)
    State = defaultdict(list)
    for outkey,outValue in output.items():
        length = 0
        StateStore = state_def(outkey)      ## 表示out[0]变量名对应的变量可能的状态
        for data in outValue[0]:
            data_insert = data[0]
            if data_insert not in StateStore.state:
                StateStore.add_state(data_insert)   ##可能需要处理*****
            StateTimeDic[length].append([outkey,data_insert]) ## [变量名，在state_def中的索引]
            length += len(data)
        State[outkey].append(StateStore)
    return State,StateTimeDic   ## State为所有变量的状态，StateTimeDic为变量在某一时刻的状态

def InputProcess(time,input):
    Condition = defaultdict(list)
    for inkey,inValue in input.items():
        length = 0
        for data in inValue[0]:
            data_insert = data[0]
            Condition[length].append([inkey,data_insert]) ## [变量名，在state_def中的索引]
            length += len(data)
    return Condition

def ConditionMatch(Transition,Index,CondMatchFlag,Condition,StateTimeDic):
    if CondMatchFlag == 2:
        CondMatchFlag = -1
        return Transition,CondMatchFlag
    else:
        if Condition[Index] and CondMatchFlag != 1:
            for Con in Condition[Index]:
                Transition.add_condition(Con)
        if StateTimeDic[Index] and CondMatchFlag != 0:
            for Element in StateTimeDic[Index]:
                Transition.add_condition(Element)
    if Transition.ConFlag == False:
        CondMatchFlag += 1
        Transition,CondMatchFlag = ConditionMatch(Transition,Index-1,CondMatchFlag,Condition,StateTimeDic)
    return Transition,CondMatchFlag

def TimerDetect(Transition,Index,StateTimeDic,time,Condition,Namelist):
    ## 可以加速，即将值一直不变的变量去除掉
    global IntervalTime
    MaxTime = 8/IntervalTime
    LoopTime = int(max(Index-MaxTime , 1))
    namelist = Namelist.copy()
    for Element in StateTimeDic[Index]:
        if Element[0] in namelist:
            namelist.remove(Element[0])
    for i in range(Index-1,LoopTime,-1):
        Time = time[Index] - time[i]
        for Element in StateTimeDic[i]:
            if Element[0] in namelist:
                namelist.remove(Element[0])
                Transition.add_timer(Element[0],Time,Element[1])
        for Element in Condition[i]:
            if Element[0] in namelist:
                namelist.remove(Element[0])
                Transition.add_timer(Element[0],Time,Element[1])
        if not namelist:
            break
    return Transition

def CombineSM(Status,StateTimeDic,Condition,time,namelist):
    CurrentState = defaultdict(list)
    length = len(time)
    for Element in StateTimeDic[0]:
        CurrentState[Element[0]].append(Element[1])
    for i in range(1,length):
        for Element in StateTimeDic[i]:
            if CurrentState[Element[0]][0] != Element[1]:
                Transition = Trans_def(CurrentState[Element[0]][0],Element[1],i)
                CondMatchFlag = 0
                Transition,CondMatchFlag = ConditionMatch(Transition,i,CondMatchFlag,Condition,StateTimeDic)
                if CondMatchFlag == -1:
                    Transition = TimerDetect(Transition,i,StateTimeDic,time,Condition,namelist)
                Status[Element[0]][0].add_transition(Transition)
                CurrentState[Element[0]][0] = Element[1]  ## 更新状态
    return Status

def ConCompare(Condition1,Key,DataFrames,Condition):
    for Con in Condition1 :
        if Con in Condition:    ##都存在的条件
            pass
        else:                   ##仅在Condition1中存在的条件
            for Data in DataFrames[1:]:
                if Data.name == Con[0]:
                    if Data.body[Key[0]] == Con[1]:
                        Condition.append(Con)
                    # elif wave_detect(Con,Key[1],DataFrames):
                    #     Condition.append([Con[0],Con[1],'wave'])
                    break
    return Condition

def wave_detect(Con,Key,DataFrames):
    for Data in DataFrames[1:]:
        if Data.name == Con[0]:
            if Data.body[Key] != Data.body[Key+1]:
                return True

def TimeCompare(Trans1,Trans2,DataFrames):  ## Transition1有Condition, Transition2没有Condition
    Key1 = Trans1.HappenTime[0]
    Key2 = Trans2.HappenTime[0]
    Timer1 = Trans1.Timer.copy()
    Timer2 = Trans2.Timer.copy()
    Condition = Trans1.Condition.copy()
    # for Con in Condition:
    #     wave_detect(Con,Key1,DataFrames)
    for Timer in Timer2:
        for Data in DataFrames[1:]:
            if Data.name == Timer[0]:
                if Data.body[Key2] == Timer[1]:
                    Timer1[Timer[0]].append([Key2,Timer[1]])
                else:
                    Timer1[Timer[0]].append([Key2,Data.body[Key2]])


def SameTransition(Trans1,Trans2,DataFrames):
    NewFlag = False
    Key1 = Trans1.HappenTime[0]
    Key2 = Trans2.HappenTime[0]
    Condition1 = Trans1.Condition.copy()
    Condition2 = Trans2.Condition.copy()
    Timer1 = Trans1.Timer.copy()
    Timer2 = Trans2.Timer.copy()
    Condition = []
    Transition = Trans_def(Trans1.SourceState,Trans1.TargetState,Key1)
    if Condition1 != Condition2:
        if Condition1 and Condition2:
            Con_Copy = Condition1.copy()
            ## 在Condition1与 Condition2中都存在的条件
            for Con1 in Con_Copy:
                if Con1 in Condition2:
                    Condition.append(Con1)
            ## 比较两个Condition中分别存在的条件，是否在另一个中满足条件
            Condition = ConCompare(Condition1,[Key2,Key1],DataFrames,Condition)
            Condition = ConCompare(Condition2,[Key1,Key2],DataFrames,Condition)
        elif not Condition1 and Condition2: ## Condition2有条件
            Condition = ConCompare(Condition2,[Key1,Key2],DataFrames,Condition)
        elif Condition1 and not Condition2: ## Condition1有条件
            Condition = ConCompare(Condition1,[Key2,Key1],DataFrames,Condition)
        if Condition != []:     ##更新条件
            Transition.Condition = Condition
            NewFlag = True
            Key = Trans1.HappenTime + Trans2.HappenTime  ##更新HappenTime
            Key = list(set(Key))
            Transition.HappenTime = Key.copy()
        else:   #全新条件
            return [],False
    else:
        Condition = Condition1.copy()
        Key = Trans1.HappenTime + Trans2.HappenTime  ##更新HappenTime
        Key = list(set(Key))
        Transition.HappenTime = Key.copy()
        NewFlag = True
    if Condition:
        Transition.Condition = Condition
    ## Timer处理&&Condition全部为空
    if not Timer1 and not Timer2:
        pass
    elif not(Condition or Timer1):
        Transition.Timer = Timer2.copy()
        Transition.TimerFlag = True
        Timer = Timer2.copy()
    elif not(Condition or Timer2):
        Transition.Timer = Timer1.copy()
        Transition.TimerFlag = True
        Timer = Timer1.copy()
    elif Timer1 and Timer2 and not Condition:
        Timer = Timer1.copy()
        Transition.TimerFlag = True
        for Key, Value in Timer1.items():
            if Key in Timer2.keys():
                Value = Value[0]
                if Value[1] == Timer2[Key][0][1]:
                    Time = Timer2[Key][0][0] - Value[0]
                    Time = Time.total_seconds()
                    if abs(Time) > 0.35:  ## 时间差大于0.4s
                        del Timer[Key]
                else:
                    del Timer[Key] ##记录
            else:
                del Timer[Key] ##记录
        Transition.Timer = Timer
        if Timer:
            NewFlag = True
            Key = Trans1.HappenTime + Trans2.HappenTime  ##更新HappenTime
            Key = list(set(Key))  ##去除重复的key
            Transition.HappenTime = Key.copy()
        else:
            NewFlag = False
    if not NewFlag:
        return [],False
    return Transition,NewFlag          

def find_time(Data, State):
    transition_vector = []
    for i in range(1, len(Data.body)-1):
        if Data.body[i+1] == State and Data.body[i] != State:
            transition_vector.append(i+1)
    return transition_vector

def nearest_disappear(KeyList,ListA):
    """
    寻找在ListA中是否有与KeyList中值的绝对值相差小于等于1的值，
    如果没有，则将该元素加入进ListB
    """
    ListB = []

    for key in KeyList:
        # 检查key是否在ListA中有相差小于等于1的值
        has_nearby_value = False
        
        for list_a_value in ListA:
            if abs(key - list_a_value) <= 1:
                has_nearby_value = True
                break
        
        # 如果没有找到相差小于等于1的值，则加入ListB
        if not has_nearby_value:
            ListB.append(key)
    
    return ListB

def Process_Counter(ListA,Key,DataFrames,namelist,IONum,StateTimeDic,Transition):
    TargetState = Transition.TargetState
    SourceState = Transition.SourceState
    best_timer_keys = None
    if Transition.Timer:
        best_timer_keys = list(Transition.Timer.keys())
    Counter_Dic = defaultdict(list)
    ListA = [0] + sorted(ListA)  # 将ListA从小到大排列
    last_index = 0
    ListProcess = []
    for Data in DataFrames[1:]:
        if Data.name == Key:
            for i in range(len(Data.body)):
                if Data.body[i] == TargetState and Data.body[i-1] == SourceState:
                    if i in ListA:
                        ListProcess.append([last_index,i])
                    elif i not in ListA:
                        last_index = i
    pass_list = []
    for Data in DataFrames[1:]:
        if Data.name == Key:
            continue
        Counter = defaultdict(list)
        for index in range(len(ListProcess)):
            State = Data.body[ListProcess[index][0]]
            for i in range(ListProcess[index][0]+1,ListProcess[index][1]):
                if State != Data.body[i]:
                    State = Data.body[i]
                    if len(Counter[State]) < index+1:
                        Counter[State].append(1)
                    else:
                        Counter[State][index] +=1
        for State,CountTime in Counter.items():
            if len(set(CountTime)) == 1:
                Counter_Dic[CountTime[0]].append([Data.name,State])
            else:
                pass_list.append([Data.name,State])
    
    counter_out = []
    NoneBoolean = defaultdict(list)
    # 按key从小到大排序读取Counter_Dic的值
    for key in sorted(Counter_Dic.keys()):
        value = Counter_Dic[key]
        for pair in value:
            if pair not in pass_list and namelist.index(pair[0]) < IONum - 1:  ## 只保留输入变量的计数
                if best_timer_keys and pair[0] in best_timer_keys:
                    continue
                if pair[1] not in [True,False]:
                    NoneBoolean[pair[0]].append(pair[1])
                else:
                    inpair = [pair[0],pair[1],key]
                    counter_out.append(inpair)
        ## 处理NoneBoolean，使得其中为范围值
        if NoneBoolean:
            counter_out = NoneBooleanProcess(NoneBoolean,DataFrames,counter_out,StateTimeDic,ListA,key)
        if counter_out:
            counter_out = counter_out[0]  ## 只保留第一个计数
            return counter_out  ## 返回的Counter_Dic是一个字典，key为计数值，value为[变量名，状态]的列表
    return None  ## 返回的Counter_Dic是一个字典，key为计数值，value为[变量名，状态]的列表

def NoneBooleanProcess(NoneBoolean,DataFrames,counter_out,StateTimeDic,ListA,times):
     ## 记录其中列表值，然后比较所有的输入值，看是否可以使用一个大于或者小于进行表示
    del ListA[0]
    listProcess = defaultdict(list)
    for key,value in NoneBoolean.items():
        if len(value) == 1:
            counter_out.append([key,value[0]])
        elif len(value) > 1:
            min_value = min(value)
            max_value = max(value)
            area_flag = [False,False,False]
            ## 如果最大值最小值超出后，则舍弃该方法
            for Data in DataFrames[1:]:
                if Data.name == key:
                    index = 0
                    for data in Data.body:
                        if data in value:
                            if StateTimeDic[index]:
                                for Element in StateTimeDic[index]:
                                    deltalist = []
                                    for x in ListA:
                                        if x >= index:
                                            deltalist.append(x - index)
                                        else:
                                            deltalist.append(float('inf'))  # 如果x小于index，则设置为-9999
                                    indexSymbol = ListA[deltalist.index(min(deltalist))]
                                    if not listProcess[data]:
                                        listProcess[data] = defaultdict(list)
                                        if indexSymbol not in listProcess[data][Element[0]]:
                                            listProcess[data][Element[0]].append(indexSymbol)
                                    else:
                                        if indexSymbol not in listProcess[data][Element[0]]:
                                            listProcess[data][Element[0]].append(indexSymbol)
                        if data > min_value and data not in value and data < max_value:
                            area_flag[1] = True
                        elif data < min_value and data not in value:
                            area_flag[0] = True
                        elif data > max_value and data not in value:
                            area_flag[2] = True
                        index += 1
            if area_flag[0] and area_flag[1] and area_flag[2]:
                flag = False
                for keya,valuea in listProcess.items():
                    for keyb,valueb in valuea.items():
                        if valueb == ListA:
                            counter_out = [key,keya,times]
                            flag = True
                            break
                    if flag:
                        break
            elif area_flag[0] and area_flag[1] and not area_flag[2]:
                counter_out.append([key,'>=' + str(max_value),times])
            elif area_flag[1] and area_flag[2] and not area_flag[0]:
                counter_out.append([key,'<=' + str(min_value),times])
            elif area_flag[0] and area_flag[2] and not area_flag[1]:
                counter_out.append([key,str(min_value) + '~' + str(max_value),times])
            elif area_flag[0] and not area_flag[1] and not area_flag[2]:
                counter_out.append([key,'>=' + str(min_value),times])
            elif area_flag[2] and not area_flag[0] and not area_flag[1]:
                counter_out.append([key,'<=' + str(max_value),times])
    return counter_out


def List_Combination(Timer_List,KeyList):
    timer_keys = list(Timer_List.keys())
    min_combination_size = float('inf')
    best_timer_keys = None
    
    # 尝试所有可能的Timer_List key组合
    for r in range(1, len(timer_keys) + 1):
        for timer_combination in combinations(timer_keys, r):
            # 获取该组合中所有Timer的列表
            combined_lists = []
            for timer_key in timer_combination:
                for timer_list in Timer_List[timer_key]:
                    combined_lists.append(timer_list)
            
            if combined_lists:
                # 计算所有列表的交集
                intersection = set(combined_lists[0])
                for timer_list in combined_lists[1:]:
                    intersection = intersection.intersection(set(timer_list))
                # 检查交集是否等于KeyList
                if intersection == set(KeyList):
                    if r < min_combination_size:
                        min_combination_size = r
                        best_timer_keys = timer_combination
                    break  # 找到了，跳出当前大小的组合
        
        # 如果找到了最小组合，不需要继续尝试更大的组合
        if best_timer_keys:
            break
    return best_timer_keys if best_timer_keys else None

def Transition_Only(Transition,DataFrames,IONum,Key,NameList,StateTimeDic):
    ## 这里需要对Transition的Condition和Timer进行唯一化
    global IntervalTime
    KeyList = []
    namelist = [Key]
    for Con in Transition.Condition:        ##缺少时间开始的条件，需要添加这一部分
        namelist.append(Con[0])
        keylist = []
        for Data in DataFrames[1:]:
            if Data.name == Con[0]:
                keylist = find_time(Data, Con[1])
        for Con1 in Transition.Condition:
            if Con != Con1:
                for Data in DataFrames[1:]:
                    if Data.name == Con1[0]:
                        for key in keylist:
                            if Data.body[key] != Con1[1]:
                                keylist.remove(key)
                        break
        for time,value in Transition.Timer.items():
            for Data in DataFrames[1:]:
                if Data.name == time:
                    for key in keylist:
                        index = key - value[0][0]
                        if Data.body[index] != value[0][1]:     ##报错时修改
                            keylist.remove(key)
                    break
        KeyList.extend(keylist)
    ListA = Transition.HappenTime 
    Timer_List = defaultdict(list)
    for time,value in Transition.Timer.items():
        namelist.append(time)
        keylist = []
        for Data in DataFrames[1:]:
            if Data.name == time:
                keylist = find_time(Data,value[0][1])
                keylist_copy = keylist.copy()
                deltatime = int((value[0][0].total_seconds()) / IntervalTime)
                # keylist = [key+deltatime for key in keylist]  ## 只保留大于等于value[0][0]的key
                keylist = []
                for index in keylist_copy:
                    indexbetter = index + deltatime
                    if indexbetter >= len(Data.body):
                        continue
                    while True:                        
                        if (DataFrames[0].body[indexbetter] - DataFrames[0].body[index]).total_seconds() <= value[0][0].total_seconds() - 0.2:
                            indexbetter +=1
                        elif (DataFrames[0].body[indexbetter] - DataFrames[0].body[index]).total_seconds() >= value[0][0].total_seconds() +0.2:
                            indexbetter -=1
                        else:
                            keylist.append(indexbetter)
                            break
                break
                
        # 将keylist中的元素与ListA进行匹配，差值小于5时替换
        corrected_keylist = []
        for key in keylist:
            best_match = key  # 默认保持原值
            min_diff = float('inf')
            
            # 查找ListA中差值最小且小于5的元素
            for list_a_key in ListA:
                diff = abs(key - list_a_key)
                if diff < 5 and diff < min_diff:
                    min_diff = diff
                    best_match = list_a_key
            
            corrected_keylist.append(best_match)
        for Con in Transition.Condition:
            if Con[0] == time:
                for key in corrected_keylist:
                    index = key - value[0][0]
                    if Data.body[index] != Con[1]:
                        corrected_keylist.remove(key)
        Timer_List[time].append(corrected_keylist)
        if KeyList == []:
            KeyList.extend(corrected_keylist)
        else:
            # 将KeyList中的元素值与corrected_keylist中的元素值相比较
            # 值的差的绝对值小于2时，则保留KeyList中该值，否则去除掉这个值
            filtered_keylist = []
            for key_val in KeyList:
                for corrected_val in corrected_keylist:
                    if abs(key_val - corrected_val) < 2:
                        filtered_keylist.append(key_val)
                        break
            # 将过滤后的KeyList与corrected_keylist合并
            KeyList = filtered_keylist
    KeyList = list(set(KeyList))  ##去除重复的key
    timer_time = 0
    best_timer_keys = None
    if Timer_List:
        best_timer_keys = List_Combination(Timer_List, KeyList)  ## 寻找最优的Timer组合
        if best_timer_keys:
            # 更新Transition的Timer，只保留最优组合的Timer
            new_timer = defaultdict(list)
            for timer_key in best_timer_keys:
                if timer_key in Transition.Timer:
                    new_timer[timer_key] = Transition.Timer[timer_key]
                    if timer_key == Key:
                        timer_time = Transition.Timer[timer_key][0][0].total_seconds()
            Transition.Timer = new_timer
            if new_timer:
                Transition.TimerFlag = True
        
    for Data in DataFrames:     ## 确保KeyList中的元素在对象变量前一时刻等于原始状态
        if Data.name == Key:
            indexs = math.ceil(timer_time*0.9 / IntervalTime)
            KeyCopy = KeyList.copy()
            for key in KeyCopy:
                if timer_time:
                    compare_list = Data.body[key-indexs:key-1]
                    compare_list_another = [Transition.SourceState] * len(compare_list)
                    if  compare_list != compare_list_another:  ## 目标状态的前一时刻等于SourceState
                        KeyList.remove(key)
                else:
                    compare_list = Data.body[key-1]
                    compare_list_another = Transition.SourceState
                    if  compare_list != compare_list_another:  ## 目标状态的前一时刻等于SourceState
                        KeyList.remove(key)
    ListB = nearest_disappear(KeyList,ListA)  ##从KeyList中找出在ListA中没有相近值的元素
    ListBcopy = ListB.copy()
    ListB = SameCompare(ListBcopy,DataFrames,Key,Transition)
    stateDic = defaultdict(list)

    ## 剩下来的就是特征
    for key in ListA:
        for Data in DataFrames[1:]:
            if Data.name not in namelist:
                if stateDic[Data.name] == []:
                    stateDic[Data.name].append(Data.body[key])
                if stateDic[Data.name][0] != Data.body[key]: # or (NameList.index(Data.name) >= IONum - 1 and Data.body[key] != Data.body[key-1]):
                    namelist.append(Data.name)
                    stateDic[Data.name] = []
    listKey = defaultdict(list)
    ListBcopy = ListB.copy()
    for key in ListBcopy:
        KeepFlag = True
        KeepList = []
        for Data in DataFrames[1:]:
            if Data.name == Key:
                if Data.body[key] == Transition.TargetState and Data.body[key-1] == Transition.SourceState:
                    KeepFlag = False
                    ListB.remove(key)  ## 只保留目标状态的转移
            if Data.name not in namelist:
                if stateDic[Data.name][0] != Data.body[key]:
                    KeepList.append([Data.name,key])
        if KeepFlag:
            for item in KeepList:
                listKey[item[0]].append(item[1])  ## 将符合条件的key添加到listKey中
    del ListBcopy
    ConditionMark = None
    if ListB:
        ConditionMark = ProcessListB(ListB,listKey,IONum,NameList)
        if ConditionMark:
            for key in ConditionMark:
                Transition.add_identify(key,stateDic[key][0])  ##添加唯一标识
        else:
            Counter = Process_Counter(ListA,Key,DataFrames,NameList,IONum,StateTimeDic,Transition);
            if Counter:
                Transition.Counter = Counter
    return Transition

def SameCompare(ListB,DataFrames,Key,Transition):
    for Data in DataFrames[1:]:
        if Data.name == Key:
            for key in ListB:
                if (Data.body[key] == Transition.TargetState and Data.body[key-1] == Transition.SourceState) or (Data.body[key] == Transition.SourceState and Data.body[key+1] == Transition.TargetState) or (Data.body[key-1] == Transition.TargetState and Data.body[key-2] == Transition.SourceState):
                    ListB.remove(key)
            break
    return ListB

def find_longest_keys(d):
    """只返回最长值对应的键"""
    if not d:
        return []
    
    max_length = max(len(value) for value in d.values())
    return [key for key, value in d.items() if len(value) == max_length]

def ProcessListB(ListB,listKey,IONum,namelist): ## 下一步处理，应该是有无限循环
    IDtt = IONum - 1
    listfin = defaultdict(list)
    while ListB:
        # 检查listKey中所有键值是否都为空
        if all(len(value) == 0 for value in listKey.values()):
            return None  # 返回没有结果
        
        inputdic = defaultdict(list)
        outputdic = defaultdict(list)
        longest_keys = find_longest_keys(listKey)
        for key in longest_keys:
            if namelist.index(key) < IDtt:
                inputdic[key].append(listKey[key])
                listKey[key] = []
            else:
                outputdic[key].append(listKey[key])
                listKey[key] = []
        if inputdic:
            for key, value in inputdic.items():
                # 从ListB中移除该key对应的元素
                listfill = value[0].copy()  # value[0]是listKey[key]
                if sorted(listfill) == sorted(ListB):
                    return [key]
                listfin[key].append(listfill)
        if outputdic:
            for key, value in outputdic.items():
                # 从ListB中移除该key对应的元素
                listfill = value[0].copy()  # value[0]是listKey[key]
                if sorted(listfill) == sorted(ListB):
                    return [key]
                listfin[key].append(listfill)
        if listfin:
            keys = list(listfin.keys())
            # 尝试所有可能的键组合
            for r in range(1, len(keys) + 1):
                if r == 1:
                    continue  # 跳过单个键的组合，因为已经处理过了
                for key_combination in combinations(keys, r):
                    # 合并选中键对应的所有列表
                    combined_list = []
                    for k in key_combination:
                        combined_list.extend(listfin[k][0])  # listfin[k][0]是该键对应的列表
                    
                    # 检查合并后的列表是否与ListB相等（忽略顺序）
                    if sorted(list(set(combined_list))) == sorted(ListB):
                        return key_combination  # 返回匹配的键组合
                    

def StateMachineCheck(Status,DataFrames,IONum,namelist,StateTimeDic):
    for Key,Value in Status.items():
        Trans = []
        Transitions = Value[0].transition
        Indexes = Value[0].SameTransition()
        if not Indexes:
            continue
        ## 合并相同转移
        for index in Indexes:
            Trans_Com = [Transitions[index[0]]]
            for i in index[1:]:
                New_Transition = []
                Flag = False
                for Trans_Judge in Trans_Com:
                    New_Transition,NewFlag = SameTransition(Trans_Judge,Transitions[i],DataFrames) ##检验两个转移是否相同
                    if New_Transition:
                        if New_Transition.Timer or New_Transition.Condition:
                            Flag = True
                            Trans_Com.remove(Trans_Judge)
                            Trans_Com.append(New_Transition)
                if not Flag:
                    Trans_Com.append(Transitions[i])
            ## 转移条件唯一化
            for Trans_Judge in Trans_Com:
                Transition = Transition_Only(Trans_Judge,DataFrames[0],IONum,Key,namelist,StateTimeDic)
                Trans.append(Transition)
        Value[0].transition = Trans
    return Status

def StateMachineGenerate(FilePath , IONum):
    DataFrames = ReadExcelFiles(FilePath)
    DataVariety = FindVariety(DataFrames,IONum)
    namelist = DataVariety.Name_List()
    Status,StateTimeDic = OutputProcess(DataVariety.time,DataVariety.output)
    Condition = InputProcess(DataVariety.time,DataVariety.input)
    StatusWithTransition = CombineSM(Status,StateTimeDic,Condition,DataVariety.time,namelist)
    Status = StateMachineCheck(StatusWithTransition,DataFrames,IONum,namelist,StateTimeDic)##合并完成
    return Status,namelist



def main():
    global IntervalTime
    IntervalTime = 0.2 ## 0.2second
    DocumentList = [ ['Distributing' , 10] , ['Measuring' , 12] , ['PickAPlace' , 10] , ['Sorting',11]]
    for DocumentInformation in DocumentList:
        FilePath = 'C:\\Users\\39330\\Desktop\\Genshin Impact\\StateMachineGenerate\\Set_01\\'+ DocumentInformation[0] +'.xlsx'
        IONum = DocumentInformation[1]
        Filename = DocumentInformation[0]
        StateMachine , namelist = StateMachineGenerate( FilePath , IONum )
        # visualize_FiniteStateMachine(StateMachine, 'fsm_output_'+str(DocumentInformation[0]))
        FilePath = 'C:\\Users\\39330\\Desktop\\Genshin Impact\\StateMachineGenerate\\Set_01\\'+ DocumentInformation[0] +'.xlsx'
        CorrectPercent = CheckCorrect(FilePath,StateMachine,IONum,namelist,Filename)
        
def visualize_FiniteStateMachine(StatusWithTransition, output_file):
    for key, value in StatusWithTransition.items():
        fsm = graphviz.Digraph(name=key, format='png')
        fsm.attr(rankdir='LR')
        # 添加状态
        for state in value[0].state:
            fsm.node(str(state))
        # 添加转移
        for transition in value[0].transition:
            source = str(transition.SourceState)
            target = str(transition.TargetState)
            Condition = []
            if transition.TimerFlag:        ## 时间Condition
                time_list = []
                for Key, Value in transition.Timer.items():
                    Value =Value[0]  ## 取第一个时间
                    formatted_time = format_timestamp_to_seconds(Value[0])
                    time_list.append(formatted_time)
                    Condition.append(Key + ' ' + formatted_time + ' ' + str(Value[1]))
                min_time = min(time_list)
                Condition = 'Timer Delay ' + str(Condition[time_list.index(min_time)])
            if transition.Condition:
                if not Condition:
                    Condition = transition.Condition[0][0] + ' ' + str(transition.Condition[0][1])
                else:
                    Condition = Condition +' & '+ transition.Condition[0][0] + ' ' + str(transition.Condition[0][1])
                if len(transition.Condition[0]) >=2:
                    for con in transition.Condition[1:]:
                        Condition = Condition + ' & ' + con[0] + ' ' + str(con[1])          
            if transition.Identify:  ## 唯一标识
                for Key, Value in transition.Identify.items():
                    Condition = Condition + ' & Symbol ' + Key + ' ' + str(Value[0])
            if transition.Counter:
                Count = transition.Counter
                Condition = Condition + ' & Counter ' + Count[0] + ' ' + str(Count[1]) + " " + str(Count[2]) + 'times'
            if Condition:
                fsm.edge(source, target, label=Condition)
            # else:
            #     fsm.edge(source, target)
        # 保存图像
        fsm.render(output_file + '_' + key)

def StateMachineSort(StateMachine):
    """
    对状态机进行排序，{Var_name:{State:[Transition1,Transition2,... ]}}
    """
    StateMachineNew = defaultdict(list)
    Timers = defaultdict(list)
    Counter = defaultdict(list)
    for var_name, state_machine in StateMachine.items():
        for state in state_machine[0].state:
            for transition in state_machine[0].transition:
                if transition.SourceState == state:
                    StateMachineNew[var_name].append(transition)
                for Key,Value in transition.Timer.items():
                    Value = Value[0]
                    if not Timers[Key]:
                        Timers[Key] = defaultdict(list)
                    Timers[Key][Value[1]] = True ## 初始化计时器
                if transition.Counter:
                    Count = transition.Counter
                    if not Counter[Count[0]]:
                        Counter[Count[0]] = defaultdict(list)
                    Counter[Count[0]][Count[1]].append([0,Count[2],var_name,transition.TargetState]) ## {varname:{state:[count,times]}}
   
    return StateMachineNew, Timers, Counter

def CheckCorrect(FilePath,StateMachine,IONum,namelist_all,Filename):
    """
    模拟执行状态机并计算正确率
    """
    global IntervalTime, accuracylow

    # try:
    DataExcelFrames = ReadExcelFiles(FilePath)
    data_frames = DataExcelFrames[0]
    length = len(data_frames[0].body)
    namelist = [Data.name for Data in data_frames[IONum:]]  # 输出变量名列表
    
    # 初始化状态变量字典 - 记录每个变量的当前状态
    StateVariety = defaultdict(list)
    OutPutValue = defaultdict(list)  # 记录输出变量的变化
    for Data in data_frames[1:]:
        if len(Data.body) >= length:
            StateVariety[Data.name] = Data.body[0]
            if Data.name in namelist:
                OutPutValue[Data.name] = [Data.body[0]]  # 初始化为包含初始值的列表
                if Data.dtype not in ['bool', 'str']:
                    StateVariety[Data.name] = StateMachine[Data.name][0].state[0]  # 初始化为状态机的初始状态
        else:
            continue
    
    # 整理状态机
    StateMachineNew,Timers,Counters = StateMachineSort(StateMachine)
    
    VarVaritey = defaultdict(list)  # 当前发生变化的输入变量
    # 逐时间步模拟执行
    varnamelist_last = []  # 记录发生变化的输出变量名
    varnamelist_now = []
    for i in range(1, length):
        current_time = data_frames[0].body[i]
        # 更新当前输入状态
        for Data in data_frames[1:IONum]:
            if StateVariety[Data.name] != Data.body[i]:#进入新的一轮，输入变量值尚未变化
                VarVaritey[Data.name] = Data.body[i]
                if Data.body[i] in Timers[Data.name]:
                    Timers[Data.name][Data.body[i]] = current_time  # 更新计时器
                if Data.body[i] in Counters[Data.name]:
                    for CountProcess in Counters[Data.name][Data.body[i]]:
                        CountProcess[0] += 1
            else:
                VarVaritey[Data.name] = []
        for var_name in namelist:       # 只检查输出变量
            StateNow = StateVariety[var_name]
            if var_name in StateMachineNew:
                for Machine in StateMachineNew[var_name]:
                    if Machine.SourceState == StateNow: ##对应的Transition
                        TransitionFlag = True
                        if Machine.Condition:
                            TransitionFlag = ConditionProcess(Machine.Condition,VarVaritey,StateVariety,var_name)
                        if Machine.TimerFlag and TransitionFlag:
                            for Key, Value in Machine.Timer.items():
                                Value = Value[0]
                                if Timers[Key][Value[1]] == True:
                                    TransitionFlag = False  # 如果计时器未启动，则跳过
                                    break  # 如果计时器未启动，则跳过
                                elif Timers[Key][Value[1]]:
                                    DeltaTime = (current_time - Timers[Key][Value[1]]).total_seconds()
                                    if abs(DeltaTime- Value[0].total_seconds()) > 0.3:  ##裕度留0.1s
                                        TransitionFlag = False
                                if not TransitionFlag:
                                    break
                        if Machine.Identify and TransitionFlag:
                            for key, value in Machine.Identify.items():
                                if VarVaritey[key] != [] and VarVaritey[key] != value[0]:
                                    TransitionFlag = False
                                    break
                                if StateVariety[key] != value[0] :
                                    TransitionFlag = False
                                    break
                                elif var_name =='Cylinder_Extend' and (VarVaritey[key] != value[0] and VarVaritey[key] != []):
                                    TransitionFlag = False
                                    break
                        if Machine.Counter and TransitionFlag:
                            count = Machine.Counter
                            for CountProcess in Counters[count[0]][count[1]]:
                                if CountProcess[2] == var_name and CountProcess[3] == Machine.TargetState:
                                    if CountProcess[0] != CountProcess[1]:
                                        TransitionFlag = False
                                        break
                        if TransitionFlag:
                            # 记录变化的输出状态
                            VarVaritey[var_name] = Machine.TargetState
                            varnamelist_now.append(var_name)
                            if Machine.Counter:
                                count = Machine.Counter
                                for CountProcess in Counters[count[0]][count[1]]:
                                    if CountProcess[2] == var_name and CountProcess[3] == Machine.TargetState:
                                        CountProcess[0] = 0  # 重置计数器
                                        break
                            # 更新计时器与计数器状态
                            if var_name in Timers and Machine.TargetState in Timers[var_name]:
                                Timers[var_name][Machine.TargetState] = current_time  # 更新计时器
                                # Timers[var_name][Machine.SourceState] = True
                            if var_name in Counters and Machine.TargetState in Counters[var_name]:
                                for CountProcess in Counters[var_name][Machine.TargetState]:
                                    CountProcess[0] += 1
        for varname1 in varnamelist_last:
            if varname1 not in varnamelist_now:
                VarVaritey[varname1] = []
        # 更新状态和输出值
        for Data in data_frames[1:]:
            var_name = Data.name
            if (VarVaritey[var_name] in [True,False,0] or VarVaritey[var_name]) and var_name in namelist:
                StateVariety[var_name] = VarVaritey[var_name]
                if var_name in namelist:
                    x = OutPutValue[var_name][-1]
                    if not isinstance(OutPutValue[var_name][-1],bool):
                        _, var = eqprocess_with_namelist(VarVaritey[var_name],OutPutValue,namelist_all,StateVariety)
                        OutPutValue[var_name].append(var)
                    else:
                        OutPutValue[var_name].append(VarVaritey[var_name])
            elif var_name not in namelist:
                StateVariety[var_name] = Data.body[i]
                VarVaritey[var_name] = []
            elif var_name in namelist:
                # 如果是输出变量但没有变化，保持当前状态
                if OutPutValue[var_name][-1] not in [True,False]:
                    _ , var = eqprocess_with_namelist(StateVariety[var_name],OutPutValue,namelist_all,StateVariety)
                    OutPutValue[var_name].append(var)
                else:
                    OutPutValue[var_name].append(OutPutValue[var_name][-1])
        varnamelist_last = varnamelist_now.copy()
        varnamelist_now = []
            # 更新当前状态为实际数据
            # if i < len(Data.body):
            #     StateVariety[var_name] = Data.body[i]
    
    # 计算正确率
    total_elements = 0
    correct_elements = 0
    
    for Data in data_frames[IONum:]:
        if Data.name in namelist:
            List1 = OutPutValue[Data.name]  # 模拟生成的输出
            List2 = Data.body           # 实际的测试数据
        
            
            # 按顺序一一比较元素
            var_total = len(List2)
            var_correct = 0
            
            for i in range(len(List1)):
                if i < len(List2) and List1[i] == List2[i]:
                    var_correct += 1
            # if var_total != var_correct:
            total_elements += var_total
            correct_elements += var_correct
            
            # 打印每个变量的详细结果
            var_accuracy = (var_correct / var_total * 100) if var_total > 0 else 100.0
            if var_accuracy < 98:
                accuracylow[Filename+Data.name] = List1
            print(f"Variable {Data.name}: {var_correct}/{var_total} correct ({var_accuracy:.2f}%)")
    
    # 计算总体正确率
    if total_elements > 0:
        overall_correctness = (correct_elements / total_elements) * 100
        print(f"Overall accuracy: {overall_correctness:.2f}%")
        print(f"========================================")
        return overall_correctness
    else:
        print("No output elements found for comparison")
        return 100.0
            

def ConditionProcess(Condition,VarVaritey,StateVariety,name):
    """
    处理状态机转移条件
    """
    for con in Condition:
        if VarVaritey[con[0]] == con[1]:  # 如果条件变量在当前输入状态中
            for con1 in Condition:
                if con != con1:
                    # 如果有多个条件满足，则进行逻辑与处理
                    if StateVariety[con1[0]] != con1[1] and VarVaritey[con1[0]] != con1[1]:
                        return False
            return True
    return False

def save_accuracylow_to_excel(filename='accuracylow_results.xlsx'):
    """
    将accuracylow字典保存到Excel文件中
    第一行为字典的key，后续行为对应的value
    """
    global accuracylow
    
    if not accuracylow:
        return
    
    # 创建一个新的工作簿
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    
    # 获取所有键
    keys = list(accuracylow.keys())
    
    # 写入第一行（键名）
    for col, key in enumerate(keys, 1):
        ws.cell(row=1, column=col, value=key)
    
    # 找出最长的值列表长度
    max_length = max(len(values) for values in accuracylow.values()) if accuracylow else 0
    
    # 写入数据
    for col, key in enumerate(keys, 1):
        values = accuracylow[key]
        for row, value in enumerate(values, 2):  # 从第2行开始
            ws.cell(row=row, column=col, value=value)
    
    # 保存文件
    wb.save(filename)
    print(f"accuracylow data saved to {filename}")


if __name__ == '__main__':
    main()