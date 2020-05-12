import quantopian.algorithm as algo
from quantopian.optimize import TargetWeights
import pandas as pd
import numpy as np
import datetime,math
from zipline.utils.calendars import get_calendar

def initialize(context):

    set_slippage( us_equities=slippage.FixedBasisPointsSlippage(basis_points=5, volume_limit=0.1),
        us_futures=slippage.FixedSlippage(spread=0)
    )
    
    set_commission(
         us_equities=commission.PerShare(cost=0.001, min_trade_cost=0),
        us_futures=commission.PerContract(cost=1, exchange_fee=0.85, min_trade_cost=0)
    )
    
    context.futures = {
    # Agricultural Commodities
    'ag_commodities': {
        'soybean': continuous_future('SY', offset=0, roll='volume', adjustment='mul'),
        'soybean_oil': continuous_future('BO', offset=0, roll='volume', adjustment='mul'),
        'soybean_meal': continuous_future('SM', offset=0, roll='volume', adjustment='mul'),
        'wheat': continuous_future('WC', offset=0, roll='volume', adjustment='mul'),
        'corn': continuous_future('CN', offset=0, roll='volume', adjustment='mul'),
        'oats': continuous_future('OA', offset=0, roll='volume', adjustment='mul'),
        'ethanol': continuous_future('ET', offset=0, roll='volume', adjustment='mul'),
        'feeder_cattle': continuous_future('FC', offset=0, roll='volume', adjustment='mul'),
        'live_cattle': continuous_future('LC', offset=0, roll='volume', adjustment='mul'),
        'lean_hogs': continuous_future('LH', offset=0, roll='volume', adjustment='mul'),
        'lumber': continuous_future('LB', offset=0, roll='volume', adjustment='mul'),
    },

    # Non-Agricultural Commodities
    'non_ag_commodities': {
    'crude_oil': continuous_future('CL', offset=0, roll='volume', adjustment='mul'),
        'natural_gas': continuous_future('NG', offset=0, roll='volume', adjustment='mul'),
        'rbob_gasoline': continuous_future('XB', offset=0, roll='volume', adjustment='mul'),
        'heating_oil': continuous_future('HO', offset=0, roll='volume', adjustment='mul'),
        'gold': continuous_future('GC', offset=0, roll='volume', adjustment='mul'),
        'silver': continuous_future('SV', offset=0, roll='volume', adjustment='mul'),
        'copper': continuous_future('HG', offset=0, roll='volume', adjustment='mul'),
        'palladium': continuous_future('PA', offset=0, roll='volume', adjustment='mul'),
        'platinum': continuous_future('PL', offset=0, roll='volume', adjustment='mul'),
    },
}
    
    # Rebalance every month.
    algo.schedule_function(
        rebalance,
        algo.date_rules.month_start(),
        algo.time_rules.market_open(),
    )
    
    # Rebalance cash position every month
    algo.schedule_function(
        func=rebalance_cash_position,
        date_rule=algo.date_rules.week_start(),
        time_rule=algo.time_rules.market_open()
        )

    # check stoploss every week
    algo.schedule_function(
        func=stop_loss,
        date_rule=algo.date_rules.week_start(),
        time_rule=algo.time_rules.market_open()) 
    
    # roll over ES future 2 days before maturity
    algo.schedule_function(
        func=roll_over,
        date_rule=algo.date_rules.every_day(),
        time_rule=algo.time_rules.market_open()) 
    
    #Fetch T-Bill 3M from Self-estabilished Google Spreadsheet
    fetch_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTbZ5aK8auK7Dlh1vY_-w1Tmsz04xTTPgW0I71cmArbeSig7w3oWVO3x-pnCFEudKKwdrHr28LOV1uM/pub?gid=0&single=true&output=csv',date_column='date', date_format='%Y-%m-%d',symbol="TB")
    
def rebalance(context, data):  
    '''
    filter tradable futures based on term structure and momentum
    @parameters
    :threshold_1: percent to filter rollYield
    :threshold_2: percent to filter excessReturn
    :look_back_period: used to compute momentum
    :chain_near: index of near contract to calculate rollYield
    :chain_distant: index of distant contract to calculate rollYield
    :volcap: portfolio volatility target level
    :vol_look_back_period: used to calculate volatility
    :stoploss_thredhold: drawdown threshold in terms of SD
    :es_look_back_period: window used to calculate ES's momentum
    :es_weight_multiple: by how much ES's weight is increased
    '''
    ########Parameter to Optimize########
    threshold_1=0.5
    threshold_2=0.4
    look_back_period=252 
    chain_near=0
    chain_distant=1
    context.maxLeverage = 1
    context.volcap = 0.2
    context.vol_look_back_period = 21 
    context.stoploss_threshold = 2  
    context.es_look_back_period = 252 
    context.es_weight_multiple = 1
    #####################################
    
    currentDate=algo.get_datetime().date()
    checkDate=currentDate+datetime.timedelta(days=-1) 
    futuresAll=context.futures
    treasuryYield=data.current("TB","yield")
    
    print("treasury Yield is %.4f"%(treasuryYield))
    
    finalFutureList=[] # contract
    finalSideList=[]
    finalcfList=[] # continuous futures
    for futureType in futuresAll:
        cfList=[]
        tradeFutureList=[]
        rollReturnList=[]
        excessReturnList=[]
        for future in futuresAll[futureType]:
            checkedFuture=futuresAll[futureType][future]
            try:
                contract_chain = data.current_chain(checkedFuture)
            except:
                print(future+" - Abnormal")
                continue

            if len(contract_chain)<=1:
                print(future+" - Incomplete")
                continue
            else:
                pass       

            try:
                NearFuture=contract_chain[chain_near]
                NearFutureEndDate=NearFuture.end_date.date() 
                NearDiffDays=(NearFutureEndDate-checkDate).days 
                NearPrice=data.history(NearFuture,'price',1,"1d").values[0]
                lnPt=math.log(NearPrice)

                DistantFuture=contract_chain[chain_distant]
                DistantFutureEndDate=DistantFuture.end_date.date() 
                DistantDiffDays=(DistantFutureEndDate-checkDate).days
                DistantPrice=data.history(DistantFuture, 'price',1,"1d").values[0]
                lnPd=math.log(DistantPrice)

                rollReturn=(lnPt-lnPd)*365/(DistantDiffDays-NearDiffDays)

                #Calculate excess return by using continuous future
                priceDF=data.history(checkedFuture, 'price',look_back_period,"1d")
                futureChange=priceDF.values[-1]/priceDF.values[0]-1            
                excessReturn=futureChange-treasuryYield
                
                #Store the return data
                cfList.append(checkedFuture)
                tradeFutureList.append(NearFuture)
                rollReturnList.append(rollReturn)
                excessReturnList.append(excessReturn)
            except:
                continue
            
    futureReturnDF=pd.DataFrame({"rollReturn":rollReturnList,               "excessReturn":excessReturnList,
      "contFuture":cfList},                   index=tradeFutureList).sort_values(by="rollReturn",
    ascending=False)
    futureReturnDF.dropna(inplace=True)
    
    count20=int(futureReturnDF.shape[0]*threshold_1)
    if count20==0:
        pass
    else:         
        buyDF=futureReturnDF.iloc[:count20,:]
        buyDF.sort_values(by="excessReturn", ascending=False,inplace=True)
        count50_buy=int(buyDF.shape[0]*threshold_2)
        buyDF=buyDF.iloc[:count50_buy,:]
        print('long futures',buyDF.index.tolist())
        for i in range(count50_buy):
            if count50_buy==0:
                continue
            else:
                finalFutureList.append(buyDF.index.tolist()[i])
                finalSideList.append(1)
                finalcfList.append(buyDF["contFuture"].tolist()[i])

        sellDF=futureReturnDF.iloc[-count20:,:]
        sellDF.sort_values(by="excessReturn", ascending=True,inplace=True)
        count50_sell=int(sellDF.shape[0]*threshold_2)
        sellDF=sellDF.iloc[:count50_sell,:]
        print('short futures', sellDF.index.tolist())
        for i in range(count50_sell):
            if count50_sell==0:
                continue
            else:                
                finalFutureList.append(sellDF.index.tolist()[i])
                finalSideList.append(-1)
                finalcfList.append(sellDF["contFuture"].tolist()[i])

    '''
    long ES future if it has positive absolute momentum
    # different signals for adding weight:
    - MA: ma1 > ma2
    - MACD: macd > signal
    '''
    ES = continuous_future('ES', offset=0, roll='calendar', adjustment='mul') # data available until 2016-12-16
    priceDF=data.history(ES, 'price',context.es_look_back_period,"1d")
    futureChange=priceDF.values[-1]/priceDF.values[0]-1  

    if futureChange > 0: 
        try:
            nearES=data.current_chain(ES)[chain_near]
            finalFutureList.insert(0,nearES)
            finalSideList.insert(0,1)
            finalcfList.insert(0,ES)
        except:
            print("SP500 Emini future - Abnormal") 
    
    # save future data
    futureDF=pd.DataFrame({"position":finalSideList,
                          "contFuture":finalcfList},
                          index=finalFutureList)
    
    if len(futureDF) == 0:
        context.cov_matrix = [np.nan]
    else:
        # df with DatetimeIndex and columns of assets
        prices = data.history(finalFutureList, 'price', context.vol_look_back_period,"1d")
        returns = prices.pct_change().dropna()
        returns.columns = map(lambda x: x.symbol, returns.columns)
        # add excessReturn in vol_look_back_period
        futureDF['return']=returns.mean(axis=0).values
        # daily variance
        cov_matrix = returns.cov()
        print('covariance matrix')
        print(cov_matrix)
        context.cov_matrix = cov_matrix.values
        
    context.futureList = futureDF.copy()
         
    # compute optimal weight
    context.weights = optimal_weight(context, data)
    print("Our positions")
    print(context.weights)
    if context.weights.shape[0]==0:
        print("All data missing, clear positions")
        for future in context.portfolio.positions.keys():
            order_target(future, 0)
    else:
        order_optimal_portfolio(TargetWeights(context.weights),
                                constraints = [])
    
    
    
def optimal_weight(context, data):
    '''
    use volatility target and risk parity to compute weight
    @params
    :cov_matrix: a numpy narray of covariance matrix 
    :futureDF: a dataframe with position, volatility and return columns
    '''
    cov_matrix = context.cov_matrix
    futureDF = context.futureList   
    num_futures = len(futureDF) 
    
    # no tradable futures
    if num_futures == 0:
        weights = futureDF.position 
        return weights
    # first equally weighted
    weights = futureDF.position / num_futures 
    weights = weights.values # change to np array

    # if long/short more than 1 futures, first perform risk-parity within each long/short portfolio
    if num_futures > 2:
        futuresVol = np.sqrt(np.diagonal(cov_matrix) * 252) 
        n_long = len(weights[weights>0])

        longVol = 1 / futuresVol[:n_long] # get reciprocal of vol
        # add es's weight
        if ('ES' in str(futureDF.index[0])):
            longVol[0] *= context.es_weight_multiple
            print('Add ES weight')
        long_w = longVol / np.sum(longVol) # risk-parity

        shortVol = 1 / futuresVol[n_long:] # get reciprocal of vol
        short_w = -1 *shortVol /  np.sum(shortVol) # risk-parity
            
        weights = np.concatenate([long_w, short_w])
        
    # volatility target
    pf_vol = np.sqrt(weights.T.dot(cov_matrix).dot(weights) * 252)
    print('portfolio volatility before volatility target is %.4f' %(pf_vol))
    weights *= min(context.volcap / pf_vol, context.maxLeverage) 
    pf_vol = np.sqrt(weights.T.dot(cov_matrix).dot(weights) * 252)
    print('portfolio volatility after volatility target is %.4f' %(pf_vol))
    weights = pd.Series(weights, index=futureDF.index)
    return weights 

def stop_loss(context, data): 
    '''
    exit pos if loss exceed stoploss_threshold * SD(daily return) over vol_look_back_period
    '''
    futureDF = context.futureList 
    cov_matrix = context.cov_matrix
    varianceDF = pd.DataFrame({'variance':cov_matrix.diagonal()},
                            index=futureDF.index)
    
    for future, position in context.portfolio.positions.items():
        if future == symbol('TLT'):
            continue
        elif future not in futureDF.index:
            continue
        # vwap paid in this position
        enter_price = position.cost_basis
        curr_price = position.last_sale_price
        pct_change = curr_price / enter_price - 1

        threshold = context.stoploss_threshold * np.sqrt(varianceDF.loc[future,'variance'])
            
        if position.amount > 0:
            if pct_change <  -threshold:
                order_target_percent(future, 0)
                print("Stop loss, exit long for %s"%(future))
                context.futureList.drop(future)
                context.weights.drop(future)
        elif position.amount < 0:
             if pct_change > threshold:
                order_target_percent(future, 0)
                print("Stop loss, cover short for %s"%(future))
                context.futureList.drop(future)
                context.weights.drop(future)      
                
def roll_over(context, data):
    '''
    roll over ES future 2 days before the contract maturity
    '''
    futureDF = context.futureList 
    EARLY_ROLL_DAYS = 2
    currentdate = get_datetime('US/Eastern')
    context.futures_calendar = get_calendar('us_futures')
 
    for pos in context.portfolio.positions:
        if 'ES' in str(pos):
            distance = context.futures_calendar.session_distance(
                currentdate, pos.auto_close_date)
            if distance == EARLY_ROLL_DAYS:
                try:
                    cf = futureDF.loc[pos,'contFuture']
                    contract = data.current_chain(cf)[1]
                    context.weights[contract] = context.weights[pos]
                    context.weights[pos] = 0
                    order_target_percent(pos,0)
                    order_target_percent(contract, context.weights[contract])
                    print('roll over for %s' %(pos))
                except:
                    print('ES Data is unavailable')
                    break
            
            
def rebalance_cash_position(context,data):
    # assume fully collateralized
    currentPosition=context.portfolio.positions.keys() 
    cash=context.portfolio.cash * 0.5
    
    if len(currentPosition)>=1 and symbol('TLT') not in currentPosition:
        order_target_value(symbol('TLT'),cash) 
    elif len(currentPosition)>=2 and symbol('TLT') in currentPosition:
        order_target_value(symbol('TLT'),cash)
    else:        
        order_target_percent(symbol('TLT'),1)
