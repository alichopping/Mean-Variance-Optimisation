import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as spo

'''Define some useful functions for downloading data, computing returns etc'''
def stockdata(tickers:list, period:str):
    stocks=yf.download(tickers, period=period)
    stocks=pd.DataFrame(stocks["Adj Close"])
    stocks=stocks.dropna(axis=0)
    return stocks #returns dataframe of Adj Close prices for each ticker over the specified period

def returns(period:str, stockdf):
    df=stockdf.resample(period).ffill().pct_change()
    return df.dropna(axis=0) #returns the returns for the stockdf for the specified period (daily, monthly etc)

def randomweights(num_portfolios, tickers):
    weights=[]
    for i in range(num_portfolios):
        row=[]
        for stock in tickers:
            w=np.random.uniform(0,1)
            row.append(w)
        weights.append(row)

    for i in range(len(weights)): #Ensure that all of the portfolio weights sum to 1
        weights[i]=np.array([weights[i][j]/sum(weights[i]) for j in range(len(weights[i]))])

    return np.array(weights) #returns a random matrix of portfolio weights, whose shape is (num_portfolios, num_assets)

def portfolio_performance(weightmatrix, num_portfolios, exp_return_vector, returns_cov_matrix):
    portfolio_return=[0]*num_portfolios
    portfolio_variance=[0]*num_portfolios
    portfolio_sharpe=[0]*num_portfolios
    
    for i in range(len(portfolio_return)):
        portfolio_return[i]=float(weightmatrix[i].transpose().dot(exp_return_vector))
        portfolio_variance[i]=weightmatrix[i].transpose().dot(returns_cov_matrix).dot(weightmatrix[i])
        portfolio_sharpe[i]=(portfolio_return[i])/np.sqrt(portfolio_variance[i])
        
    return np.array(portfolio_return), np.array(portfolio_variance), np.array(portfolio_sharpe) #returns vectors of expected return, variance and sharpe ratio for each portfolio weighting


def portfolio_returns(weights, exp_return_vector):
    return weights.transpose().dot(exp_return_vector)
    
    
def min_portfolio_variance(weights, exp_return_vector, returns_cov_matrix):
    return weights.transpose().dot(returns_cov_matrix).dot(weights)


#-------------------------------------------------------------------------------------------------------------------
#Download some data, compute daily returns
tickers=["SPY", "QQQ", "AMZN", "AAPL", "HES", "KO", "MRK"]
stocks=stockdata(tickers, period="4y")
daily_returns=returns("D", stocks)

#Compute expected log return for each asset and annualize it by multiplying by 252
#Compute a covariance matrix from the daily returns and again, annualize it.
alpha=pd.DataFrame()#alpha is a dataframe of expected annualised log returns for each stock. 1 column, len(tickers) rows.

for stock in list(daily_returns.columns):
    log_returns = np.log(1 + daily_returns[stock])  #Compute log returns
    annualized_log_return = log_returns.mean() * 252  #Annualise
    alpha.loc[stock, "Expected Annualized Return"] = annualized_log_return

sigma=daily_returns.cov()*252 #sigma is a len(tickers)xlen(tickers) covariance matrix, indicated the covariances between the returns of each stock.
    

#Compute matrix of random portfolio weights and compute the expected return, vol and sharpe for each portfolio
num_portfolios=70000
weightmatrix=randomweights(num_portfolios, tickers)

alphaw, sigmaw, sharpe=portfolio_performance(weightmatrix, num_portfolios, alpha, sigma)


#Scatter plot of all portfolios resulting from the random weight matrix
plt.figure(figsize=(16,8))
plt.scatter(sigmaw, alphaw, s=50, alpha=0.5, c=sharpe)
plt.title("Portfolio Variance vs Log Return", size=15)
plt.xlabel("Portfolio Variance", size=15)
plt.ylabel("Expected Portfolio Log Return", size=15)
plt.grid(visible=True)
plt.colorbar(label="Sharpe Ratio")
plt.show()


#Plot efficient frontier; create an array of target returns and find the smallest variances for those targets.
#Now we're minimising the variance subject to two constraints; that the weights sum to 1
#and that the expected return is equal to the target return
optimal_variances=[]
optimal_weights=[]
target_returns=np.linspace(alphaw.min(), alphaw.max(), 50) #target returns is a linspace between the smallest expected portfolio return and the largest
bounds = tuple((0,1) for x in range(len(tickers))) #The weights are required to be bounded between 0 and 1
initial_guess=[1/len(tickers)]*len(tickers) #Start with an equal-weighted portfolio

for target_return in target_returns:
    constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1}, #minimising a function of weights x, subject to constraints
                   {'type' : 'eq', 'fun': lambda x: portfolio_returns(x, alpha)[0] - target_return})
    optimal_variance=spo.minimize(min_portfolio_variance,
                                     initial_guess,
                                     method = 'SLSQP',
                                     bounds = bounds,
                                     constraints = constraints,
                                     args=(alpha, sigma))
    optimal_variances.append(optimal_variance.fun)
    optimal_weights.append(optimal_variance.x)#optimal_weights[i] is the set of weights corresponding to optimal_variances[i]


#Scatter plot of all portfolios + feasible set
plt.figure(figsize=(16,8))
plt.plot(optimal_variances, target_returns, linestyle="--", linewidth=3, color = "black", label="Feasible Set")
plt.scatter(sigmaw, alphaw, s=50, alpha=0.5, c=sharpe)
plt.title("Feasible Set", size=15)
plt.xlabel("Portfolio Variance", size=15)
plt.ylabel("Expected Portfolio Log Return", size=15)
plt.grid(visible=True)
plt.colorbar(label="Sharpe Ratio")
plt.legend(loc="upper left", prop={'size': 15})
plt.show()


#Scatter plot of all portfolios + efficient frontier
min_feasible_var=np.argmin(optimal_variances)
frontier_vars=optimal_variances[min_feasible_var:]
frontier_returns=target_returns[min_feasible_var:]
frontier_weights=optimal_weights[min_feasible_var:]

plt.figure(figsize=(16,8))
plt.plot(frontier_vars, frontier_returns, linestyle="--", linewidth=3, color = "black", label="Efficient Frontier")
plt.scatter(sigmaw, alphaw, s=50, alpha=0.25, c=sharpe, cmap="gray")
plt.title("Efficient Frontier", size=15)
plt.xlabel("Portfolio Variance", size=15)
plt.ylabel("Expected Portfolio Log Return", size=15)
plt.grid(visible=True)
plt.colorbar(label="Sharpe Ratio")
plt.legend(loc="upper left", prop={'size': 15})
plt.show()


#Plot performance of a randomly-selected portfolio compared to that of a portfolio with the
#same risk, but on the efficient frontier
random_index=np.random.choice(range(num_portfolios))
random_weights=weightmatrix[random_index]
random_portfolio_variance=sigmaw[random_index]
random_portfolio_returns=random_weights.dot(daily_returns.transpose())

closest_frontier_var=min(frontier_vars, key=lambda x:abs(x-random_portfolio_variance))
frontier_index=frontier_vars.index(closest_frontier_var)
frontier_portfolio_weights=np.array(frontier_weights[frontier_index])
frontier_portfolio_returns=frontier_portfolio_weights.dot(daily_returns.transpose())

plt.figure(figsize=(16,8))
plt.plot(daily_returns.index, frontier_portfolio_returns.cumsum(), label=f"Efficient Portfolio, $\sigma^2\sim${round(closest_frontier_var,3)}")
plt.plot(daily_returns.index, random_portfolio_returns.cumsum(), label=f"Random Portfolio, $\sigma^2\sim${round(random_portfolio_variance,3)}", color="gray")
plt.title("Performance: Efficient Portfolio vs Random Portfolio of Comparable Variance", size=15)
plt.xlabel("Time", size=15)
plt.ylabel("Cumulative Returns", size=15)
plt.grid(visible=True)
plt.legend(loc="upper left", prop={'size': 15})
plt.show()



