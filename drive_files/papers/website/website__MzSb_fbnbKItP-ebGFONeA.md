# WorldQuant 101 Formulaic Alphas — 五大经典因子详解

来源: 微信公众号
URL: https://mp.weixin.qq.com/s/MzSb_fbnbKItP-ebGFONeA
提取时间: 2026-07-04 22:09:00

## 提取摘要
本文详细解析 WorldQuant 101 Formulaic Alphas 论文中的五个经典量化因子：动量反转5/20日均线比、量价相关20日、波动率20日、价格偏离MA20、量能反转5日。文章介绍了每个因子的通俗解释、数学公式、对应101 Alphas原型以及实战含义和失效场景。

## 构造思路
五个因子均为日频单股票计算，使用标准价量字段（close, volume, returns），无需截面操作，适用于daily_single模板。
