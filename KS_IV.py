# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:30:00 2019

@author: lxt
"""

import math

import numpy as np
import pandas as pd

class KS():
    def __init__(self, label="label", k=10, n_split=8, output_path="./",
                 output_if=True, ks_max=True, iv_sum=True):
        """
        Parameters
        ----------
        label : str, default "label"
            标签列名
        k : int or str, default 10
            最终结果分段的段数
        n_split : int ot str, default 10
            用于区分训练集中单列的去重特征值个数以评定为连续型变量或者离散型变量，
            大于等于 8 则该列判为连续型变量，小于则为离散型变量
        output_path : str, default './'
            KS分析结果输出路径，输出文件名为 KS结果.xlsx
        output_if : bool, default True
            是否输出KS分析结果文档, True 则输出， False 则直接返回结果
        ks_max : bool, defalut True
            是否输出各个特征KS最大值
        iv_sum : bool, default True
            是否输出各个特征IV值
        """
        self.label = label
        self.k = int(k)
        self.n_split = n_split
        self.ks_max = ks_max
        self.iv_sum = iv_sum
        self.output_path = output_path
        self.output_if = output_if
        self.jd_true_fea = []

    def judge(self, data, col):
        """判断是连续变量还是离散变量"""
        if data[col].value_counts().shape[0] >= self.n_split:
            return True
        elif data[col].value_counts().shape[0] == 0:
            return False
        else:
            return False

    def judge_dtypes(self, data, col):
        """判断是数值型变量还是文本型变量"""
        try:
            data[col].astype(float)
            return True
        except:
            return False

    def util(self, res):
        """对于生成ks结果做汇总求和处理并便于合并"""
        res["分数段"] = res["分数段"].astype(str)
        usr_sum = res["用户数"].sum()  # 用户数总数
        usr_overdue_sum = res["逾期用户数"].sum()  # 逾期用户数总数
        overdue_rate = usr_overdue_sum / usr_sum if usr_sum != 0 else np.nan  # 总逾期率
        ks_m =  res["KS"].max()  # 最大 KS
        iv_s = res["IV"].sum()   # IV 总和
        res.loc[len(res), ["用户数","逾期用户数","逾期率","KS","IV"]] = [
                usr_sum, usr_overdue_sum, overdue_rate, ks_m, iv_s]
        res.loc[len(res), :] = [None] * res.shape[1]  # 加一行空白行
        res.loc[len(res), :] = res.columns.tolist() # 加一行表头与下一个表拼接
        return res, ks_m, iv_s

    def section(self, data, col, label, k, continuous=True, runtype=False):
        """特征值分段函数
        Parameters
        ----------
        data : dataframe.
            输入数据
        col : str.
            需要计算ks的列名
        continuous : bool.
            是否是连续型变量
        runtype : bool.
            调用接口，当 True 时返回 user_count 以及 df。当 False 时返回 user_count

        Return
        ------
        user_count : dataframe. sectiontion result
        df : dataframe. the column data
        qt : section
        """
        if continuous:
            s = np.arange(0, 1, 1/k) # 生成 1/k 等差数列
            #计算对应列的分位数对应的值
            qt = data[col].quantile(s, interpolation = 'lower')

            val_max = data[col].max()
            qt = qt.tolist()
            if val_max not in qt:
                qt.append(val_max) #由于arange不包含尾部，因此新增最大值到q

            qt = list(set(qt))
            qt.sort()

            df = pd.DataFrame(data[col].tolist())

            # 划分分数段
            df['分数段'] = pd.cut(df[0], qt, include_lowest = True)
            df = df.join(data[label])
            user_count = df.groupby('分数段').size()  #计算用户数
        else:
            df = data[[col, label]]

            # 划分分数段
            df["分数段"] = data[col]
            user_count = df[col].value_counts()  #计算用户数
            qt = user_count.values.tolist()

        if runtype is True:
            return user_count, df, qt  # 返回已便后续使用
        else:
            return user_count  # 单独调用 section 函数时， 只返回 user_count

    def ks(self, data, col):
        """ ks计算函数

        Parameters
        ----------
        data: dataframe. a dataframe of data
        col: str. the feature column in data

        Return
        ------
        temp: dataframe. the result
        """
        jd = self.judge(data, col) # 判断 该列是否是连续变量

        user_count, df, qt = self.section(data, col, self.label, self.k,
                                          continuous=jd, runtype=True)

        user_count.name = '用户数' # 把 user_count 转化为 dataframe:temp
        temp = user_count.reset_index().rename(columns={"index":"分数段"})
        temp = temp.sort_values("分数段", ascending=True)
        temp.insert(0, '字段', col)
        temp['用户占比'] = temp['用户数']/temp['用户数'].sum()  #用户占比

        # 某分段可能没有逾期用户，因此需要分数段 merge 关联
        tmp_overdue = df[df[self.label] == 1].groupby('分数段').size()
        tmp_overdue.name = "逾期用户数"
        tmp_overdue = tmp_overdue.reset_index().rename(columns={"index":"分数段"})
        temp = temp.merge(tmp_overdue, how="left", on=["分数段"])
        temp["逾期用户数"] = temp["逾期用户数"].fillna(0)

        temp['逾期率'] = temp['逾期用户数']/temp['用户数']  #计算逾期率
        temp['好用户数'] = temp['用户数'] - temp['逾期用户数']  #好用户数
        temp['逾期用户占比'] = temp['逾期用户数']/temp['逾期用户数'].sum()
        temp['好用户数占比'] = temp['好用户数']/temp['好用户数'].sum()  
        temp['累计逾期用户'] = temp['逾期用户数'].cumsum() #累计逾期客户
        temp['累计好用户数'] = temp['好用户数'].cumsum()  #累计好客户
        temp['累计逾期客户占比'] = temp['累计逾期用户']/temp['逾期用户数'].sum()  #累计逾期客户占比
        temp['累计好客户占比'] = temp['累计好用户数']/temp['好用户数'].sum()  #累计好客户占比
        temp['WOE'] = [math.log(
            (temp['逾期用户数'][q]/temp['好用户数'][q])/
            (temp['逾期用户数'].sum()/temp['好用户数'].sum())
            ) if temp['逾期用户数'][q] != 0 else 0
            for q in range(len(temp['分数段']))] #计算WOE
        temp['IV'] = [(temp['逾期用户数'][q]/temp['逾期用户数'].sum()-
            temp['好用户数'][q]/temp['好用户数'].sum())*temp['WOE'][q]
            if temp['逾期用户数'][q] != 0 else 0 for q in range(len(temp['分数段']))]
        temp['KS']= (temp['累计好客户占比'] - temp['累计逾期客户占比']).abs()  #KS值计算
        temp['下限'] = qt[:-1] if jd else temp['分数段']
        temp['上限'] = qt[1:] if jd else temp['分数段']
        tmp_col = ['字段','分数段','下限','上限','用户数','用户占比','逾期用户数',
                   '逾期率','好用户数','累计逾期用户','累计好用户数','累计好客户占比',
                   '累计逾期客户占比','KS','WOE','IV']   #重新排列列
        temp = temp.loc[:, tmp_col]
        return temp

    def output(self, df_grp, df_ks_m, df_iv_s):

        excel_writer = pd.ExcelWriter(self.output_path + "/KS结果.xlsx")
        df_grp.to_excel(excel_writer, encoding="utf-8", index=False,
                        sheet_iame="KS")
        if self.ks_max:
            df_ks_m.to_excel(excel_writer, encoding="utf-8", index=True,
                             sheet_iame="KS_MAX")
        if self.iv_sum:
            df_iv_s.to_excel(excel_writer, encoding="utf-8", index=True,
                             sheet_iame="IV_SUM")

        excel_writer.save()
        excel_writer.close()

    def get(self, data):

        if data is None or data.empty:
            return "数据异常，跳出"

        data = data.reset_index(drop=True) # 重置数据的索引，避免后续计算混乱

        data_columns = [col for col in data.columns.tolist() if col != self.label]

        res = []
        ks_dic = {}
        iv_dic = {}
        data = data.where(data != "", None) # 空字符串转成 None

        for col in data_columns:
            if self.judge_dtypes(data, col):
                self.jd_true_fea.append(col)  # 收集连续型变量
                res_tmp = self.ks(data, col)  # 计算 KS
                res_tmp = self.util(res_tmp)
                res.append(res_tmp[0])

                if self.ks_max:
                    ks_dic[col] = res_tmp[1]  # KS max

                if self.iv_sum:
                    iv_dic[col] = res_tmp[2]  # IV sum

        df_grp = pd.concat(res)  # 将 KS 数据框 聚合

        df_ks_m = pd.Series(ks_dic)  # 转为数据框
        df_ks_m.name = "ks_max"
        df_ks_m = df_ks_m.to_frame()

        df_iv_s = pd.Series(iv_dic)  # 转为数据框
        df_iv_s.name = "iv_sum"
        df_iv_s = df_iv_s.to_frame()

        if self.output_if:
            self.output(df_grp, df_ks_m, df_iv_s)
        else:
            return [df_grp, df_ks_m, df_iv_s]
        false_fea = [col for col in data.columns.tolist()
            if col not in self.jd_true_fea and col != self.label] # 未计算的文本型特征
        print("已输出结果文件至 {} 下\n本次运行特征共 {} 个\n成功计算特征 {} 个\n"
              "未计算特征 {} 个\n\n".format(self.output_path, data.shape[0],
                     len(self.jd_true_fea), len(false_fea)))
