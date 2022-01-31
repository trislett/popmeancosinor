#!/usr/bin/env python

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests, fdrcorrection
from scipy.stats import t, f
from joblib import Parallel, delayed
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from tfce_mediation.pyfunc import convert_fs
from scipy.stats import circmean, circstd, norm
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

def number_to_array(variable):
	if isinstance(variable, int):
		variable = np.array([variable])
	if isinstance(variable, float):
		variable = np.array([variable])
	return(np.array(variable))

def label_to_surface(labels, values, masked = None):
	outvalues = np.array(values)
	if masked is not None:
		outvalues[masked] = -1
	outdata = np.zeros_like(labels, dtype=float)
	for i, value in enumerate(outvalues):
		label_num = i+1
		index_arr = labels == label_num
		outdata[index_arr] = value
	outdata = outdata[:, np.newaxis, np.newaxis]
	return outdata.astype(np.float32, order = "C")

def generate_seeds(n_seeds, maxint = int(2**32 - 1)):
	return([np.random.randint(0, maxint) for i in range(n_seeds)])

def fwer_corrected_p(permuted_arr, target, right_tail_probability = True, apply_fwer_correction = True):
	"""
	Calculates the FWER corrected p-value
	
	Parameters
	----------
	permuted_arr : array
		Array of permutations [N_permutations, N_factors]
	target : array or float
		statistic(s) to check against null array
	right_tail_probability : bool
		Use right tail distribution (default: True)
	apply_fwer_correction : bool
		If True, output the family-wise error rate across all factors, else output permuted p-value for each factors' distribution (default: True)
	Returns
	---------
	pval_corrected : array
		Family-wise error rate corrected p-values or permuted p-values
	"""
	if permuted_arr.ndim == 1:
		permuted_arr = permuted_arr.reshape(-1,1)
	if isinstance(target, float):
		target = np.array([target])
	assert target.ndim == 1, "Error: target array must be 1D array or float"
	n_perm, n_factors = permuted_arr.shape
	if apply_fwer_correction: 
		permuted_arr = permuted_arr.max(1)
		pval_corrected = np.divide(np.searchsorted(np.sort(permuted_arr), target), n_perm)
	else:
		if n_factors == 1:
			pval_corrected = np.divide(np.searchsorted(np.sort(permuted_arr), target), n_perm)
		else:
			assert n_factors == target.shape[0], "Error: n_factors must equal length of target for elementwise comparison"
			pval_corrected = np.zeros_like(target)
			for i in range(n_factors):
				pval_corrected[i] = np.divide(np.searchsorted(np.sort(permuted_arr[:,i]), target[i]), n_perm)
	if right_tail_probability:
		pval_corrected = 1 - pval_corrected
	return(pval_corrected)



class population_mean_cosinor():
	def __init__(self, times, y, groups, period = 24.0, n_jobs = 12, n_permutations = 10000, n_bootstraps = 10000, scale_y_by_group = False, alpha = 0.05):
		"""
		Initialize the mean function
		"""
		self.times_ = np.array(times)
		self.groups_ = np.array(groups)
		self.ugroups_ = np.unique(groups)
		self.period_ = number_to_array(period)
		self.n_period_ = len(self.period_)
		self.exog = self._dmy_code_cosine_exog(self.times_, self.period_)
		self.X = np.column_stack(self.exog)
		if scale_y_by_group:
			y_temp = []
			for group in np.unique(groups):
				y_temp.append(scale(y[groups == group]))
			self.y_ = np.concatenate(y_temp)
		else:
			self.y = np.array(y)
		# automatically adjust dimensionality
		if self.y.ndim == 1:
			self.y = self.y[:, np.newaxis]
		self.n_targets_ = self.y.shape[1]
		self.n_jobs = n_jobs
		self.n_permutations = n_permutations
		self.n_bootstraps = n_bootstraps
		self.alpha = alpha
		# additional variables
		self.N = len(self.groups_)
		self.K = len(self.ugroups_)
		self.df1 = 2*self.n_period_
		self.df2 = self.K - self.df1

	def predict(self, X, add_popmesor = False, convert_time = False):
		"""
		Return predicted endogenous variables based on the population mean coefficients
		"""
		assert hasattr(self,'mean_coef_'), "Error: Run fit_pop_mean"
		if convert_time:
			X = np.column_stack(self._dmy_code_cosine_exog(X, self.period_))
		yhat = np.dot(X, self.mean_coef_.T)
		if add_popmesor:
			yhat += self.mean_mesor_
		return(yhat)

	def calculate_pop_mean_CI(self, pop_coefs, component, y = None, alpha = 0.05):
		"""
		Bingham et al. 1982
		Formulas: Covariance estimate [59], Zero amplitude test[63], Acrophase CI [64], Amplitude CI [65] 
		"""
		period = self.period_[component]
		if pop_coefs.ndim == 3: 
			pop_coefs = pop_coefs[:,:,(component*2):(component*2 + 2)]
		else:
			pop_coefs = pop_coefs[:,(component*2):(component*2 + 2)]
		mean_coef = np.mean(pop_coefs, 0)

		if y is None:
			# formula [64] from Bingham et al. 1982
			dBetaSqr = np.zeros((self.n_targets_))
			dGammaSqr = np.zeros((self.n_targets_)) 
			dbetaGamma = np.zeros((self.n_targets_))
			for i in range(pop_coefs.shape[1]):
				db2, dbg, _, dg2 = np.cov(pop_coefs[:,i,0], pop_coefs[:,i,1]).flatten()
				dBetaSqr[i] = db2
				dGammaSqr[i] = dg2
				dbetaGamma[i] = dbg
			r_values = dbetaGamma / (np.sqrt(dBetaSqr) * np.sqrt(dGammaSqr))

			# this will be convenient for multiple components in the future
			Amplitude2 = self.pop_amplitude_[component]**2
			Beta = mean_coef[:,0]
			Beta2 = mean_coef[:,0]**2
			Gamma = mean_coef[:,1]
			Gamma2 = mean_coef[:,1]**2
			# get the SDs from the covariance matrix like in Bingham et al. 1982
			dBeta = dBetaSqr**0.5
			dBeta2 = dBetaSqr
			dGamma = dGammaSqr**0.5
			dGamma2 = dGammaSqr

			# Zero amplitude test for significance
			# Formula [63] from Bingham et al. 1982
			# I kept screwing this up, so I copied from https://rdrr.io/cran/cosinor2/src/R/cosinor2.R
			part1 = (self.K*(self.K-2))/(2*(self.K-1))
			part2 = 1/(1 - r_values**2)
			part3 = Beta2 / dBeta2
			part4 = (Beta*Gamma)/(dBeta*dGamma)
			part5 = Gamma2 / dGamma2
			brack = part3 - (2*r_values*part4) + part5
			
			F_amplitude_targets_ = part1 * part2 * brack
			pval_amplitude_targets_ = 1 - f.cdf(F_amplitude_targets_, self.df1, self.df2)
			qval_amplitude_targets_ = fdrcorrection(pval_amplitude_targets_)[1]


			C22 = np.divide((dBeta2*Beta2 + 2*dbetaGamma*Gamma*Beta + dGamma2*Gamma2), (self.K*Amplitude2))
			C23 = np.divide((-1*(dBeta2-dGamma2)*(Beta*Gamma)) + (dbetaGamma*(Beta2-Gamma2)), (self.K*Amplitude2))
			C33 = np.divide((dBeta2*Gamma2) - (2*dbetaGamma*Beta*Gamma) + (dGamma2*Beta2), (self.K*Amplitude2))

			pop_amplitude_upper_ = self.pop_amplitude_[component] + self.tcrit_*np.sqrt(C22)
			pop_amplitude_lower_ = self.pop_amplitude_[component] - self.tcrit_*np.sqrt(C22)
			temp = np.arctan(((C23 *(self.tcrit_**2))+((self.tcrit_*np.sqrt(C33))*np.sqrt((self.pop_amplitude_[component]**2)-(((C22*C33)-(C23**2))*((self.tcrit_**2)/C33)))))/((self.pop_amplitude_[component]**2)-(C22*(self.tcrit_**2))))
			temp[pop_amplitude_lower_ < 0] = np.nan
			pop_acrophase_upper_ = self.pop_acrophase_[component] - temp
			pop_acrophase_lower_ = self.pop_acrophase_[component] + temp
			pop_acrophase_time_upper_ = np.abs(pop_acrophase_upper_/(2*np.pi)) * period
			pop_acrophase_time_upper_[pop_acrophase_time_upper_>period] -= period
			pop_acrophase_time_lower_ = np.abs(pop_acrophase_lower_/(2*np.pi)) * period
			pop_acrophase_time_lower_[pop_acrophase_time_lower_>period] -= period
			if component == 0:
				self.F_amplitude_targets_ = F_amplitude_targets_
				self.pval_amplitude_targets_ = pval_amplitude_targets_
				self.qval_amplitude_targets_ = qval_amplitude_targets_
				self.pop_amplitude_lower_ = pop_amplitude_lower_
				self.pop_amplitude_upper_ = pop_amplitude_upper_
				self.pop_acrophase_lower_ = pop_acrophase_lower_
				self.pop_acrophase_upper_ = pop_acrophase_upper_
				self.pop_acrophase_time_lower_ = pop_acrophase_time_lower_
				self.pop_acrophase_time_upper_ = pop_acrophase_time_upper_
			else:
				self.F_amplitude_targets_ = np.column_stack((self.F_amplitude_targets_, F_amplitude_targets_))
				self.pval_amplitude_targets_ = np.column_stack((self.pval_amplitude_targets_, pval_amplitude_targets_))
				self.qval_amplitude_targets_ = np.column_stack((self.qval_amplitude_targets_, qval_amplitude_targets_))
				self.pop_amplitude_lower_ = np.column_stack((self.pop_amplitude_lower_,pop_amplitude_lower_))
				self.pop_amplitude_upper_ = np.column_stack((self.pop_amplitude_upper_,pop_amplitude_upper_))
				self.pop_acrophase_lower_ = np.column_stack((self.pop_acrophase_lower_,pop_acrophase_lower_))
				self.pop_acrophase_upper_ = np.column_stack((self.pop_acrophase_upper_,pop_acrophase_upper_))
				self.pop_acrophase_time_lower_ = np.column_stack((self.pop_acrophase_time_lower_,pop_acrophase_time_lower_))
				self.pop_acrophase_time_upper_ = np.column_stack((self.pop_acrophase_time_upper_,pop_acrophase_time_upper_))
		else:
			tcrit_ = abs(t.ppf((alpha/2), df=self.K-1))
			n_targets = y.ndim
			if n_targets == 1:
				y = y[:, np.newaxis]
				mean_coef = mean_coef.reshape(1,-1)
			amplitude, acrophase, acrophase_time = self._coef_to_cosinor_metrics(mean_coef, period)

			if n_targets == 1:
				dBetaSqr, dbetaGamma, _, dGammaSqr = np.cov(pop_coefs[:,0], pop_coefs[:,1]).flatten()
				r_values = dbetaGamma / (np.sqrt(dBetaSqr) * np.sqrt(dGammaSqr))
			else:
				dBetaSqr = np.zeros((n_targets))
				dGammaSqr = np.zeros((n_targets))
				dbetaGamma = np.zeros((n_targets))
				for i in range(pop_coefs.shape[1]):
					db2, dbg, _, dg2 = np.cov(pop_coefs[:,i,0], pop_coefs[:,i,1]).flatten()
					dBetaSqr[i] = db2
					dGammaSqr[i] = dg2
					dbetaGamma[i] = dbg

			Amplitude2 = amplitude**2
			Beta = mean_coef[:, 0]
			Beta2 = mean_coef[:, 0]**2
			Gamma = mean_coef[:, 1]
			Gamma2 = mean_coef[:, 1]**2
			dBeta = dBetaSqr**0.5
			dBeta2 = dBetaSqr
			dGamma = dGammaSqr**0.5
			dGamma2 = dGammaSqr

			# Zero amplitude test for significance
			# Formula [63] from Bingham et al. 1982
			# I kept screwing this up, so I copied from https://rdrr.io/cran/cosinor2/src/R/cosinor2.R
			part1 = (self.K*(self.K-2))/(2*(self.K-1))
			part2 = 1/(1 - r_values**2)
			part3 = Beta2 / dBeta2
			part4 = (Beta*Gamma)/(dBeta*dGamma)
			part5 = Gamma2 / dGamma2
			brack = part3 - (2*r_values*part4) + part5
			
			F_amp = part1 * part2 * brack
			pval_amp = 1 - f.cdf(F_amp, self.df1, self.df2)
			qval_amp = fdrcorrection(pval_amp)[1]

			# Formulas [64, 65] from Bingham et al. 1982
			C22 = np.divide((dBeta2*Beta2 + 2*dbetaGamma*Gamma*Beta + dGamma2*Gamma2), (self.K*Amplitude2))
			C23 = np.divide((-1*(dBeta2-dGamma2)*(Beta*Gamma)) + (dbetaGamma*(Beta2-Gamma2)), (self.K*Amplitude2))
			C33 = np.divide((dBeta2*Gamma2) - (2*dbetaGamma*Beta*Gamma) + (dGamma2*Beta2), (self.K*Amplitude2))

			amplitude_upper_ = amplitude + tcrit_*np.sqrt(C22)
			amplitude_lower_ = amplitude - tcrit_*np.sqrt(C22)

			temp = np.arctan(((C23 *(tcrit_**2))+((tcrit_*np.sqrt(C33))*np.sqrt((amplitude**2)-(((C22*C33)-(C23**2))*((tcrit_**2)/C33)))))/((amplitude**2)-(C22*(tcrit_**2))))
			temp[amplitude_lower_ < 0] = np.nan

			acrophase_upper_ = acrophase - temp
			acrophase_lower_ = acrophase + temp
			acrophase_time_upper_ = np.abs(acrophase_upper_/(2*np.pi)) * period
			acrophase_time_upper_[acrophase_time_upper_>period] -= period
			acrophase_time_lower_ = np.abs(acrophase_lower_/(2*np.pi)) * period
			acrophase_time_lower_[acrophase_time_lower_>period] -= period

			return(F_amp, pval_amp, qval_amp,
						amplitude, amplitude_lower_, amplitude_upper_,
						acrophase, acrophase_lower_, acrophase_upper_,
						acrophase_time, acrophase_time_lower_, acrophase_time_upper_)

	def _bootstrapCI(self, i, period, split = 0.5, subject_mesor = None, subject_coef = None, seed = None):
		if subject_coef is None:
			subject_coef = self.subject_coef_
		if subject_mesor is None:
			subject_mesor = self.subject_mesor_
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		subindex = np.arange(0, self.K, 1)
		split_size = int(split*self.K)
		bindex = np.random.permutation(subindex)[:split_size]
		bcoef = subject_coef[bindex].mean(0)
		bmesor = subject_mesor[bindex].mean(0)
		bamp, bacr, bacrtime = self._coef_to_cosinor_metrics(bcoef, period)
		return(bmesor,bamp, bacr, bacrtime)

	def _compare_acrophases(self, subset_model_significant = True):
		if subset_model_significant:
			combos = combinations(np.argwhere(self.qval_model_targets_ < 0.05)[:,0], 2)
			n_combos = len(list(combos))
			combos = combinations(np.argwhere(self.qval_model_targets_ < 0.05)[:,0], 2)  #iter tools weirdness
		else:
			combos = combinations(range(self.n_targets_), 2)
			n_combos = len(list(combos))
			combos = combinations(range(self.n_targets_), 2)
		tvalues = np.zeros((n_combos))
		absmdvalues = np.zeros((n_combos))
		corrmat = np.zeros((self.n_targets_, self.n_targets_))
		contrast = []
		loc = 0
		for i,j in combos: 
			tvalues[loc] = np.divide((self.bootstrap_CI_acrophase_mean_[i] - self.bootstrap_CI_acrophase_mean_[j]), np.sqrt(np.divide(self.bootstrap_CI_acrophase_std_[i] + self.bootstrap_CI_acrophase_std_[j], self.K)))
			absmdvalues[loc] = np.abs(self.bootstrap_CI_acrophase_mean_[i] - self.bootstrap_CI_acrophase_mean_[j])
			contrast.append("%d_%d" % (i,j))
			corrmat[i,j] = np.abs(self.bootstrap_CI_acrophase_mean_[i] - self.bootstrap_CI_acrophase_mean_[j])
			corrmat[j,i] = np.abs(self.bootstrap_CI_acrophase_mean_[i] - self.bootstrap_CI_acrophase_mean_[j])
			loc += 1
		contrast = np.array(contrast)
		pval = 2*(1 - t.cdf(abs(tvalues), (self.K-1)))
		qval = fdrcorrection(pval)[1]
		return(tvalues, absmdvalues, contrast, pval, qval, corrmat)

	def fit_pop_mean(self, fit_mesor = True, bootstrap_group = True, fit_summary_targets = True, fit_pca = False):
		"""
		Fit the population mean model
		"""
		self.fit_mesor = fit_mesor
		m_pop = []
		coefs = []
		y_demesor = []
		for group in self.ugroups_:
			lm = LinearRegression(fit_intercept=True).fit(self.X[self.groups_ == group], self.y[self.groups_ == group])
			m_pop.append(lm.intercept_)
			coefs.append(lm.coef_)
			y_demesor.append(self.y[self.groups_ == group] - lm.intercept_)
		m_pop = np.array(m_pop)
		coefs_pop = np.array(coefs)
		# write out metrics
		self.y_demesor_ = np.concatenate(y_demesor)
		self.subject_coef_ = np.array(coefs_pop)
		self.subject_mesor_ = np.array(m_pop)
		self.mean_mesor_ = np.mean(m_pop, 0)
		self.mean_coef_ = np.mean(coefs_pop, 0)
		self.sd_mesor_ = np.std(m_pop, 0)
		self.sd_coef_ = np.std(coefs_pop, 0)
		self.se_mesor_ = np.divide(np.std(m_pop, 0), np.sqrt(self.K-1))
		self.se_coef_ = np.divide(np.std(coefs_pop, 0), np.sqrt(self.K-1))
		self.tcrit_ = abs(t.ppf((self.alpha/2), df=self.K-1))

		self.pval_popmesor_ = 2 * (1 - t.cdf(abs(np.divide(self.mean_mesor_, self.se_mesor_)), self.K-1))
		self.qval_popmesor_ = fdrcorrection(self.pval_popmesor_)[1]
		
		self.lower_CI_popmesor_ = self.mean_mesor_ - ((self.tcrit_*self.sd_mesor_)/((self.K)**0.5))
		self.upper_CI_popmesor_ = self.mean_mesor_ + ((self.tcrit_*self.sd_mesor_)/((self.K)**0.5))

		self.pval_popcoef_ = np.zeros_like(self.mean_coef_)
		self.qval_popcoef_ = np.zeros_like(self.mean_coef_)
		self.lower_CI_popcoef_ = np.zeros_like(self.mean_coef_)
		self.upper_CI_popcoef_ = np.zeros_like(self.mean_coef_)
		for b in range(self.mean_coef_.shape[1]):
			self.pval_popcoef_[:,b] = 2 * (1 - t.cdf(abs(np.divide(self.mean_coef_[:,b], self.se_coef_[:,b])), self.K-1))
			self.qval_popcoef_[:,b] =  fdrcorrection(self.pval_popcoef_[:,b])[1]
			self.lower_CI_popcoef_[:,b] = self.mean_coef_[:,b] - ((self.tcrit_*self.sd_coef_[:,b])/((self.K-1)**0.5))
			self.upper_CI_popcoef_[:,b] = self.mean_coef_[:,b] + ((self.tcrit_*self.sd_coef_[:,b])/((self.K-1)**0.5))
		pop_amplitude = []
		pop_acrophase = []
		pop_acrophase_time = []
		for p in range(self.n_period_):
			percoef = np.array(self.mean_coef_[:,(p*2):(p*2 + 2)])
			pm_amp, pm_acr, pm_acr24 = self._coef_to_cosinor_metrics(percoef, self.period_[p])
			pop_amplitude.append(pm_amp)
			pop_acrophase.append(pm_acr)
			pop_acrophase_time.append(pm_acr24)
		self.pop_amplitude_ = np.array(pop_amplitude)
		self.pop_acrophase_ = np.array(pop_acrophase)
		self.pop_acrophase_time_ = np.array(pop_acrophase_time)

		if fit_mesor:
			self.ytrue_ = np.array(self.y)
		else:
			self.ytrue_ = np.array(self.y_demesor_)
		self.yhat_ = self.predict(X = self.X, add_popmesor = False)
		if fit_mesor:
			for g, group in enumerate(self.ugroups_):
				self.yhat_[self.groups_ == group] = self.yhat_[self.groups_ == group] + m_pop[g]

		for c in range(self.n_period_):
			self.calculate_pop_mean_CI(pop_coefs = coefs_pop, component = c)
			if bootstrap_group:
				period = self.period_[c]
				seeds = generate_seeds(self.n_bootstraps)
				bmesor, bamplitude, bacrophase, bacrophasetime = zip(*[self._bootstrapCI(i, period = period, subject_coef = self.subject_coef_[:,:,(c*2):(c*2 + 2)], split = 0.5, seed = None) for i in range(self.n_bootstraps)])
				bootstrap_CI_mesor_upper_ = np.quantile(bmesor, 0.95, axis = 0) 
				bootstrap_CI_mesor_lower_ = np.quantile(bmesor, 0.05, axis = 0)
				bootstrap_CI_amplitude_upper_ = np.quantile(bamplitude, 0.95, axis = 0) 
				bootstrap_CI_amplitude_lower_ = np.quantile(bamplitude, 0.05, axis = 0) 
				
				self.temp = np.array(bacrophase)
				bootstrap_CI_mesor_mean_ = np.mean(bmesor,0)
				bootstrap_CI_amplitude_mean_ = np.mean(bamplitude,0)
				bootstrap_CI_acrophase_mean_ = circmean(bacrophase, low = -2*np.pi, high=0, axis = 0)
				bootstrap_CI_acrophase_std_ = circstd(bacrophase, low = -2*np.pi, high=0, axis = 0)
				acro_error_CI = np.divide(norm.ppf(.975)*bootstrap_CI_acrophase_std_, np.sqrt(self.K * 0.5)) # alpha = 0.05, N = n_subject * split size
				bootstrap_CI_acrophase_upper_ = bootstrap_CI_acrophase_mean_ - acro_error_CI
				bootstrap_CI_acrophase_lower_ = bootstrap_CI_acrophase_mean_ + acro_error_CI

				bootstrap_CI_acrophase_time_mean_ = circmean(bacrophasetime, low = 0, high=24, axis = 0)
				bootstrap_CI_acrophase_time_std_ = circstd(bacrophasetime, low = 0, high=24, axis = 0)
				acrotime_error_CI = np.divide(norm.ppf(.975)*bootstrap_CI_acrophase_time_std_, np.sqrt(self.K * 0.5))

				bootstrap_CI_acrophase_time_upper_ = bootstrap_CI_acrophase_time_mean_ + acrotime_error_CI
#				bootstrap_CI_acrophase_time_upper_[bootstrap_CI_acrophase_time_upper_>period] -= period
				bootstrap_CI_acrophase_time_lower_ = bootstrap_CI_acrophase_time_mean_ - acrotime_error_CI
#				bootstrap_CI_acrophase_time_lower_[bootstrap_CI_acrophase_time_lower_>period] -= period
				if c == 0:
					self.bootstrap_CI_mesor_mean_ = bootstrap_CI_mesor_mean_
					self.bootstrap_CI_amplitude_mean_ = bootstrap_CI_amplitude_mean_
					self.bootstrap_CI_acrophase_mean_ = bootstrap_CI_acrophase_mean_
					self.bootstrap_CI_acrophase_std_ = bootstrap_CI_acrophase_std_
					self.bootstrap_CI_mesor_upper_ = bootstrap_CI_mesor_upper_
					self.bootstrap_CI_mesor_lower_ = bootstrap_CI_mesor_lower_
					self.bootstrap_CI_amplitude_upper_ = bootstrap_CI_amplitude_upper_
					self.bootstrap_CI_amplitude_lower_ = bootstrap_CI_amplitude_lower_
					self.bootstrap_CI_acrophase_upper_ = bootstrap_CI_acrophase_upper_
					self.bootstrap_CI_acrophase_lower_ = bootstrap_CI_acrophase_lower_
					self.bootstrap_CI_acrophase_time_upper_ = bootstrap_CI_acrophase_time_upper_
					self.bootstrap_CI_acrophase_time_lower_ = bootstrap_CI_acrophase_time_lower_
				else:
					self.bootstrap_CI_mesor_mean_ = np.column_stack((self.bootstrap_CI_mesor_mean_, bootstrap_CI_mesor_mean_))
					self.bootstrap_CI_amplitude_mean_ = np.column_stack((self.bootstrap_CI_amplitude_mean_, bootstrap_CI_amplitude_mean_))
					self.bootstrap_CI_acrophase_mean_ = np.column_stack((self.bootstrap_CI_acrophase_mean_, bootstrap_CI_acrophase_mean_))
					self.bootstrap_CI_acrophase_std_ = np.column_stack((self.bootstrap_CI_acrophase_std_, bootstrap_CI_acrophase_std_))
					self.bootstrap_CI_mesor_upper_ = np.column_stack((self.bootstrap_CI_mesor_upper_, bootstrap_CI_mesor_upper_))
					self.bootstrap_CI_mesor_lower_ = np.column_stack((self.bootstrap_CI_mesor_lower_, bootstrap_CI_mesor_lower_))
					self.bootstrap_CI_amplitude_upper_ = np.column_stack((self.bootstrap_CI_amplitude_upper_,bootstrap_CI_amplitude_upper_))
					self.bootstrap_CI_amplitude_lower_ = np.column_stack((self.bootstrap_CI_amplitude_lower_,bootstrap_CI_amplitude_lower_))
					self.bootstrap_CI_acrophase_upper_ = np.column_stack((self.bootstrap_CI_acrophase_upper_,bootstrap_CI_acrophase_upper_))
					self.bootstrap_CI_acrophase_lower_ = np.column_stack((self.bootstrap_CI_acrophase_lower_,bootstrap_CI_acrophase_lower_))
					self.bootstrap_CI_acrophase_time_upper_ = np.column_stack((self.bootstrap_CI_acrophase_time_upper_,bootstrap_CI_acrophase_time_upper_))
					self.bootstrap_CI_acrophase_time_lower_ = np.column_stack((self.bootstrap_CI_acrophase_time_lower_,bootstrap_CI_acrophase_time_lower_))

		# calculate least squares metrics
		self.MSS_ = np.sum((self.yhat_ - self.ytrue_.mean(0))**2, 0)
		self.RSS_ = np.sum((self.ytrue_ - self.yhat_)**2, 0)
		self.TSS_ = np.sum((self.ytrue_ - np.mean(self.ytrue_,0))**2,0)
		self.F_model_targets_ = np.divide(np.divide(self.MSS_, self.df1), np.divide(self.RSS_, (self.N - self.df1)))
		self.R2_targets_ = 1 - self.RSS_/self.TSS_
		self.R2_ = np.mean(self.R2_targets_)
		self.R2_adj_targets_ = 1 - np.divide((1-self.R2_targets_)*(self.N-1), (self.N - self.df1))
		self.RMSE_targets_ = np.sqrt(self.RSS_)
		self.RMSE_ = np.mean(self.RMSE_targets_)
		self.pval_model_targets_ = 1 - f.cdf(self.F_model_targets_, self.df1, self.N - self.df1)
		self.qval_model_targets_ = fdrcorrection(self.pval_model_targets_)[1]
#		self.R2_model_ = r2_score(ytrue, yhat)
#		self.R2_model_targets_ = r2_score(ytrue, yhat, multioutput='raw_values')

		if fit_summary_targets:
			if fit_pca:
				pca_ = PCA()
				self.ysumvar_ = pca_.fit_transform(scale(self.y))[:,0]
				self.pca_explained_variance_ratio_ = pca_.explained_variance_ratio_[0]
				self.pca_model_ = pca_
			else:
				self.ysumvar_ = self.y.mean(1)
			pop_mesor = []
			pop_coefs = []
			ydemesor = []
			for group in self.ugroups_:
				lm = LinearRegression(fit_intercept=True).fit(self.X[self.groups_ == group], self.ysumvar_[self.groups_ == group])
				pop_mesor.append(lm.intercept_)
				pop_coefs.append(lm.coef_)
				ydemesor.append(self.ysumvar_[self.groups_ == group] - lm.intercept_)
			ydemesor = np.concatenate(ydemesor)
			self.ydemesor = ydemesor
			pop_coefs = np.array(pop_coefs)
			pop_mesor = np.array(pop_mesor)
			self.sumvar_subject_coef_ = pop_coefs
			self.sumvar_mean_coef_ = np.mean(pop_coefs, 0)
			sumvar_pop_amplitude = []
			sumvar_pop_acrophase = []
			sumvar_pop_acrophase_time = []
			for c in range(self.n_period_):
				sumvar_percoef = np.array(self.sumvar_mean_coef_[(c*2):(c*2 + 2)])
				pm_amp, pm_acr, pm_acr24 = self._coef_to_cosinor_metrics(sumvar_percoef.reshape(1,-1), self.period_[c])
				Fval, pval, qval, amp, amp_l, amp_u, acr, acr_l, acr_u, acr_t, acr_t_l, acr_t_u = self.calculate_pop_mean_CI(pop_coefs = pop_coefs, component = c, y = self.ysumvar_)
				if c ==0:
					self.sumvar_F_amplitude_ = Fval
					self.sumvar_pval_amplitude_ = pval
					self.sumvar_pop_amplitude_ = amp
					self.sumvar_pop_amplitude_lower_ = amp_l
					self.sumvar_pop_amplitude_upper_ = amp_u
					self.sumvar_pop_acrophase_ = acr
					self.sumvar_pop_acrophase_lower_ = acr_l
					self.sumvar_pop_acrophase_upper_ = acr_u
					self.sumvar_pop_acrophase_time_ = acr_t
					self.sumvar_pop_acrophase_time_lower_ = acr_t_l
					self.sumvar_pop_acrophase_time_upper_ = acr_t_u
				else:
					self.sumvar_F_amplitude_ = np.column_stack((self.sumvar_F_amplitude_, Fval))
					self.sumvar_pval_amplitude_ = np.column_stack((self.sumvar_pval_amplitude_, pval))
					self.sumvar_pop_amplitude_ = np.column_stack((self.sumvar_pop_amplitude_, amp))
					self.sumvar_pop_amplitude_lower_ = np.column_stack((self.sumvar_pop_amplitude_lower_, amp_l))
					self.sumvar_pop_amplitude_upper_ = np.column_stack((self.sumvar_pop_amplitude_upper_, amp_u))
					self.sumvar_pop_acrophase_ = np.column_stack((self.sumvar_pop_acrophase_, acr))
					self.sumvar_pop_acrophase_lower_ = np.column_stack((self.sumvar_pop_acrophase_lower_, acr_l))
					self.sumvar_pop_acrophase_upper_ = np.column_stack((self.sumvar_pop_acrophase_upper_, acr_u))
					self.sumvar_pop_acrophase_time_ = np.column_stack((self.sumvar_pop_acrophase_time_, acr_t))
					self.sumvar_pop_acrophase_time_lower_ = np.column_stack((self.sumvar_pop_acrophase_time_lower_, acr_t_l))
					self.sumvar_pop_acrophase_time_upper_ = np.column_stack((self.sumvar_pop_acrophase_time_upper_, acr_t_u))

	def results_csv(self, csv_basename, target_name_array = None):
		assert hasattr(self,'R2_'), "Error: Run fit_pop_mean"
		pdOUT = pd.DataFrame()
		if target_name_array is None:
			pdOUT["Target"] = np.arange(1, self.n_targets_+1, 1)
		else:
			pdOUT["Target"] = target_name_array
		pdOUT["R-sqr"] = self.R2_targets_
		pdOUT["R-sqr-adj"] = self.R2_adj_targets_
		pdOUT["RMSE"] = self.RMSE_targets_
		pdOUT["Fmodel_%d_%d" % (int(self.df1), int(self.N - self.df1))] = self.F_model_targets_
		pdOUT["Pvalue"] = self.pval_model_targets_
		pdOUT["Qvalue"] = self.qval_model_targets_
		pdOUT["Pop-MESOR"] = self.mean_mesor_
		pdOUT["Pop-MESOR-Pvalue"] = self.pval_popmesor_
		pdOUT["Pop-MESOR-Qvalue"] = self.qval_popmesor_
		pdOUT["Pop-MESOR-lowerCI"] = self.lower_CI_popmesor_
		pdOUT["Pop-MESOR-upperCI"] = self.upper_CI_popmesor_
		if hasattr(self,'bootstrap_CI_mesor_upper_'):
			pdOUT["Boot-MESOR-lowerCI"] = self.bootstrap_CI_mesor_lower_
			pdOUT["Boot-MESOR-upperCI"] = self.bootstrap_CI_mesor_upper_
		if self.n_period_ == 1:
			pdOUT["Fpop_%d_%d" % (self.df1,self.df2)] = self.F_amplitude_targets_
			pdOUT["Pvalue-pop"] = self.pval_amplitude_targets_
			pdOUT["Qvalue-pop"] = self.qval_amplitude_targets_
			pdOUT["Pop-Amplitude"] = self.pop_amplitude_[0]
			pdOUT["Pop-Amplitude-lowerCI"] = self.pop_amplitude_lower_
			pdOUT["Pop-Amplitude-upperCI"] = self.pop_amplitude_upper_
			if hasattr(self,'bootstrap_CI_mesor_upper_'):
				pdOUT["Boot-Amplitude-lowerCI"] = self.bootstrap_CI_amplitude_lower_
				pdOUT["Boot-Amplitude-upperCI"] = self.bootstrap_CI_amplitude_upper_
			pdOUT["Pop-Acrophase"] = self.pop_acrophase_[0]
			pdOUT["Pop-Acrophase-lowerCI"] = self.pop_acrophase_lower_
			pdOUT["Pop-Acrophase-upperCI"] = self.pop_acrophase_upper_
			if hasattr(self,'bootstrap_CI_mesor_upper_'):
				pdOUT["Boot-Acrophase-lowerCI"] = self.bootstrap_CI_acrophase_lower_
				pdOUT["Boot-Acrophase-upperCI"] = self.bootstrap_CI_acrophase_upper_
			pdOUT["Pop-AcrophaseTime"] = self.pop_acrophase_time_[0]
			pdOUT["Pop-AcrophaseTime-lowerCI"] = self.pop_acrophase_time_lower_
			pdOUT["Pop-AcrophaseTime-upperCI"] = self.pop_acrophase_time_upper_
			if hasattr(self,'bootstrap_CI_mesor_upper_'):
				pdOUT["Boot-AcrophaseTime-lowerCI"] = self.bootstrap_CI_acrophase_time_lower_
				pdOUT["Boot-AcrophaseTime-upperCI"] = self.bootstrap_CI_acrophase_time_upper_
		else:
			for c in range(self.n_period_):
				pdOUT["Fpop%d_%d_%d" % ((c+1), self.df1,self.df2)] = self.F_amplitude_targets_[c]
				pdOUT["Pvalue-pop%d" % (c+1)] = self.pval_amplitude_targets_[c]
				pdOUT["Qvalue-pop%d" % (c+1)] = self.qval_amplitude_targets_[c]
				pdOUT["Pop-Amplitude%d" % (c+1)] = self.pop_amplitude_[0,c]
				pdOUT["Pop-Amplitude%d-lowerCI" % (c+1)] = self.pop_amplitude_lower_[c]
				pdOUT["Pop-Amplitude%d-upperCI" % (c+1)] = self.pop_amplitude_upper_[c]
				if hasattr(self,'bootstrap_CI_mesor_upper_'):
					pdOUT["Boot-Amplitude%d-lowerCI" % (c+1)] = self.bootstrap_CI_amplitude_lower_[c]
					pdOUT["Boot-Amplitude%d-upperCI" % (c+1)] = self.bootstrap_CI_amplitude_upper_[c]
				pdOUT["Pop-Acrophase%d" % (c+1)] = self.pop_acrophase_[0,c]
				pdOUT["Pop-Acrophase%d-lowerCI" % (c+1)] = self.pop_acrophase_lower_[c]
				pdOUT["Pop-Acrophase%d-upperCI" % (c+1)] = self.pop_acrophase_upper_[c]
				if hasattr(self,'bootstrap_CI_mesor_upper_'):
					pdOUT["Boot-Acrophase%d-lowerCI" % (c+1)] = self.bootstrap_CI_acrophase_lower_[c]
					pdOUT["Boot-Acrophase%d-upperCI" % (c+1)] = self.bootstrap_CI_acrophase_upper_[c]
				pdOUT["Pop-AcrophaseTime%d" % (c+1)] = self.pop_acrophase_time_[0,c]
				pdOUT["Pop-AcrophaseTime%d-lowerCI" % (c+1)] = self.pop_acrophase_time_lower_[c]
				pdOUT["Pop-AcrophaseTime%d-upperCI" % (c+1)] = self.pop_acrophase_time_upper_[c]
				if hasattr(self,'bootstrap_CI_mesor_upper_'):
					pdOUT["Boot-AcrophaseTime%d-lowerCI" % (c+1)] = self.bootstrap_CI_acrophase_time_lower_[c]
					pdOUT["Boot-AcrophaseTime%d-upperCI" % (c+1)] = self.bootstrap_CI_acrophase_time_upper_[c]
		pdOUT.to_csv("Results_Targets_%s.csv" % csv_basename, index=None)


	def plot_summary_variable(self, png_basename = None):
		assert hasattr(self,'sumvar_pop_acrophase_upper_'), "Error: Run fit_pop_mean with option fit_run_pca"
		line = 0
		if self.n_period_ == 1:
			line = line + self._predict_cosinor(np.arange(-0.1,24.1, 0.1), 0, self.sumvar_pop_amplitude_[0], self.sumvar_pop_acrophase_[0], period = self.period_[0])
			line_lower = line - (self.sumvar_pop_amplitude_upper_[0] - self.sumvar_pop_amplitude_[0])
			line_upper = line + (self.sumvar_pop_amplitude_upper_[0] - self.sumvar_pop_amplitude_[0])
		else:
			line_lower = 0
			line_upper = 0
			for c in range(self.n_period_):
				line = line + self._predict_cosinor(np.arange(-0.1,24.1, 0.1), 0, self.sumvar_pop_amplitude_[0,c], self.sumvar_pop_acrophase_[0,c], period = self.period_[c])
				line_lower = line_lower - (self.sumvar_pop_amplitude_upper_[0,c] - self.sumvar_pop_amplitude_[0,c])
				line_upper = line_upper + (self.sumvar_pop_amplitude_upper_[0,c] - self.sumvar_pop_amplitude_[0,c])
			line_lower +=line
			line_upper +=line

		plt.plot(np.arange(-0.1,24.1, 0.1), line, color = 'b', markersize=4)
		plt.scatter(self.times_, self.ydemesor, s = 4, marker = 'o', color = 'k')

		lineX = np.column_stack(self._dmy_code_cosine_exog(np.arange(-0.1,24.1, 0.1), period = self.period_))
		
		for i in range(len(self.sumvar_subject_coef_)):
			plt.plot(np.arange(-0.1,24.1, 0.1), (np.dot(lineX, self.sumvar_subject_coef_[i].T)), color = 'grey', ls = 'dotted', markersize=1, alpha = 0.5)
		plt.fill_between(np.arange(-0.1,24.1, 0.1), line_lower, line_upper, alpha=0.5, color = 'cornflowerblue', edgecolor = 'b')
		plt.xticks(np.arange(0,25, 1), range(25))
		plt.axvspan(-0.1, 6, color = 'grey', alpha = 0.2)
		plt.axvspan(20, 24.1, color = 'grey', alpha = 0.2)
		if self.n_period_ == 1:
			plt.axvline(self.sumvar_pop_acrophase_time_[0], ls = 'dashed', color = 'k')
			plt.axvline(self.sumvar_pop_acrophase_time_upper_[0], ls = 'dashed', color = 'k', alpha = 0.6)
			plt.axvline(self.sumvar_pop_acrophase_time_lower_[0], ls = 'dashed', color = 'k', alpha = 0.6)
		if png_basename is not None:
			plt.savefig("%s_summary_variable_fit.png" % png_basename)
			plt.close()
		else:
			plt.show()

	def plot_bootstrap_acrophase(self, png_basename = None):
		plt.figure(figsize=(6, 12))
		sort_idx = np.argsort(self.pop_acrophase_time_[0])
		significance = self.qval_model_targets_[sort_idx]
		sacr = np.array(self.pop_acrophase_time_[0][sort_idx])
		sacr_err = np.array(self.bootstrap_CI_acrophase_time_upper_[sort_idx]) - sacr
		for i in range(self.n_targets_):
			if significance[i] < 0.05:
				plt.errorbar(sacr[i], i, xerr=sacr_err[i], marker='s', color = 'k', alpha = 1.0)
			else:
				plt.errorbar(sacr[i], i, xerr=sacr_err[i], marker='s', color = 'grey', alpha = 0.5)
		plt.axvspan(-0.1, 6, color = 'grey', alpha = 0.2)
		plt.axvspan(20, 24.1, color = 'grey', alpha = 0.2)
		plt.xticks(np.arange(0,25, 1), range(25))
		plt.xlabel("Time (h)")
		plt.tight_layout()
		plt.xlim(-0.1, 24.1)
		if png_basename is not None:
			plt.savefig("%s_acrophase_bootstrap_error.png" % png_basename)
			plt.close()
		else:
			plt.show()
		

	def plot_circular(self, png_basename = None, set_r_to_neglogp = False, set_r_to_r2 = False):
		x = np.arange(0, 2*np.pi, np.pi/4)
		x_labels = list(map(lambda i: 'T ' + str(i) + " ", list((x/(2*np.pi) * self.period_[0]).astype(int))))
		x_labels[1::2] = [""]*len(x_labels[1::2])

		fig = plt.figure()
		ax = fig.add_subplot(projection='polar', theta_offset=(np.pi/2), theta_direction = -1)
		lines = []
		if set_r_to_r2:
			rvar = self.R2_targets_
			ax.set_rmax(np.max(rvar))
		elif set_r_to_neglogp:
			rvar = -np.log10(self.pval_amplitude_targets_)
			ax.set_rmax(np.max(rvar))
		else:
			rvar = MinMaxScaler().fit_transform(self.pop_amplitude_.T)[:,0]
			ax.set_rmax(1)
		for i in range(len(self.pop_acrophase_time_[0])):
			if self.qval_model_targets_[i] < 0.05:
				ax.annotate("", xy=(-self.pop_acrophase_[0,i], rvar[i]), xytext=(0, 0), color = 'b', arrowprops=dict(arrowstyle="->", alpha = 0.75, linewidth=2))
			else:
				ax.annotate("", xy=(-self.pop_acrophase_[0,i], rvar[i]), xytext=(0, 0), color = 'grey', arrowprops=dict(arrowstyle="-|>", alpha = 0.25, linewidth=2))

		ax.set_rticks([0.5])  # Less radial ticks
		ax.set_yticklabels([""])
		ax.set_xticks(x)
		ax.set_xticklabels(x_labels)
		ax.grid(True)
		ax.set_facecolor('#f0f0f0')
		if png_basename is not None:
			if set_r_to_r2:
				plt.savefig("%s_rsqr_circular_plot.png" % png_basename)
			elif set_r_to_neglogp:
				plt.savefig("%s_neglogp_circular_plot.png" % png_basename)
			else:
				plt.savefig("%s_amp_circular_plot.png" % png_basename)
			plt.close()
		else:
			plt.show()

#	def _permute_amplitude(self, p, seed):
#		if seed is None:
#			np.random.seed(np.random.randint(4294967295))
#		else:
#			np.random.seed(seed)
#		perm_coefs = []
#		for group in self.ugroups_:
#			pX = np.random.permutation(self.X[self.groups_ == group])
#			permlm = LinearRegression(fit_intercept=True).fit(pX, self.y[self.groups_ == group])
#			perm_coefs.append(permlm.coef_)
#		perm_mean_coef_ = np.mean(perm_coefs, 0)
#		perm_amp = []
#		for p in range(self.n_period_):
#			per_coef = perm_mean_coef_[:,(p*2):(p*2 + 2)]
#			perm_amp.append(np.sqrt((per_coef[:,0]**2) + (per_coef[:,1]**2)))
#		return(np.array(perm_amp))

#	def _permute_popmodel(self, p, seed = None):
#		if seed is None:
#			np.random.seed(np.random.randint(4294967295))
#		else:
#			np.random.seed(seed)
#		perm_m_pop = []
#		perm_coefs = []
#		perm_X = []
#		for group in self.ugroups_:
#			pX = np.random.permutation(self.X[self.groups_ == group])
#			perm_X.append(pX)
#			permlm = LinearRegression(fit_intercept=True).fit(pX, self.y[self.groups_ == group])
#			perm_m_pop.append(permlm.intercept_)
#			perm_coefs.append(permlm.coef_)
#		perm_X = np.concatenate(perm_X)
#		perm_mean_mesor_ = np.mean(perm_m_pop, 0)
#		perm_mean_coef_ = np.mean(perm_coefs, 0)
#		perm_yhat = np.dot(perm_X, perm_mean_coef_.T)
#		if self.fit_mesor:
#			perm_yhat += perm_mean_mesor_
#			ytrue = np.array(self.y)
#		else:
#			ytrue = np.array(self.y_demesor_)
#		perm_r2_model = r2_score(perm_yhat, ytrue)
#		perm_r2_model_targets = r2_score(perm_yhat, ytrue, multioutput='raw_values')
#		return(perm_r2_model, perm_r2_model_targets)

#	def permute_popop_cosinor(self):
#		seeds = generate_seeds(self.n_permutations)
#		output = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(delayed(self._permute_popmodel)(p = p, seed = seeds[p]) for p in range(self.n_permutations))
#		permR2, permR2targets = zip(*output)
#		self.perm_R2_model_ = np.array(permR2)
#		self.perm_R2_model_targets_ = np.array(permR2targets)

#	def permute_popop_cosinor_amplitude(self):
#		seeds = generate_seeds(self.n_permutations)
#		permAmplitude = np.array(Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(delayed(self._permute_amplitude)(p = p, seed = seeds[p]) for p in range(self.n_permutations)))
#		self.perm_amplitude_ = np.array(permAmplitude).swapaxes(0,1)

	def _coef_to_cosinor_metrics(self, coef, period):
		amplitude = np.sqrt((coef[:,0]**2) + (coef[:,1]**2))
		acrophase = np.arctan(np.abs(np.divide(-coef[:,1], coef[:,0])))
		acrophase[(coef[:,1] > 0) & (coef[:,0] >= 0)] = -acrophase[(coef[:,1] > 0) & (coef[:,0] >= 0)]
		acrophase[(coef[:,1] > 0) & (coef[:,0] < 0)] = (-1*np.pi) + acrophase[(coef[:,1] > 0) & (coef[:,0] < 0)]
		acrophase[(coef[:,1] < 0) & (coef[:,0] <= 0)] = (-1*np.pi) - acrophase[(coef[:,1] < 0) & (coef[:,0] <= 0)]
		acrophase[(coef[:,1] <= 0) & (coef[:,0] > 0)] = (-2*np.pi) + acrophase[(coef[:,1] <= 0) & (coef[:,0] > 0)]
		acrophase_time = np.abs(acrophase/(2*np.pi)) * period
		acrophase_time[acrophase_time>period] -= period
		return(amplitude, acrophase, acrophase_time)

	def _set_cosinor(time = None, period = None):
		"""
		Change cosinor model setting and recalculate exog
		"""
		if times is not None:
			self.times_ = np.array(times)
		if period is not None:
			self.period_ = number_to_array(period)
		self.exog = self._dmy_code_cosine_exog(self.times_, self.period_)

	def _dmy_code_cosine_exog(self, time, period = 24.0):
		"""
		Dummy codes a time variable into a cosine
		C1 = cos(2.0*Pi*time)/period)
		C2 = sin(2.0*Pi*time)/period)

		Parameters
		----------
		time : array
			1D array variable of any type 
		period : float
			Defined period (i.e., for one entire cycle) for the time variable

		Returns
		---------
		exog_arr : array
			cosine exog array (N_periods, exogs)
		
		"""
		time = np.array(time)
		period = number_to_array(period)
		exog_arr = []
		for p, per in enumerate(period):
			exog = np.cos(np.divide(2.0*np.pi*time, per))
			exog = np.column_stack((exog, np.sin(np.divide(2.0*np.pi*time, per))))
			exog_arr.append(exog)
		return(np.array(exog_arr))
		
	def _predict_cosinor(self, time_variable, mesor, amp, acr, period = 24.0):
		period = number_to_array(period)
		try:
			time_variable_ = np.ones((len(time_variable), len(mesor))) * time_variable.reshape(-1,1)
		except:
			time_variable_ = np.array(time_variable)
		yhat = mesor
		for p, per in enumerate(period):
			if period.shape[0] > 1:
				amp_c = amp[p]
				acr_c = acr[p]
			else:
				amp_c = np.squeeze(amp)
				acr_c = np.squeeze(acr)
			yhat = yhat + (amp*np.cos((2*np.pi*time_variable_)/per + acr))
		return(yhat)

