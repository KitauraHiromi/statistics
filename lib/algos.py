def binary_search(_func, _y, _upper, _lower, _err):
	# _func should be monotonically increasing or decreasing
	upper = _upper
	lower = _lower
	while upper > lower:
		mid = (upper + lower) / 2
		y = func(mid)
		if abs(y - _y) < err:
			return mid
		elif (_func(mid) - _y)*(_func(lower) - _y) < 0:
			upper = mid
		else:
			lower = mid
