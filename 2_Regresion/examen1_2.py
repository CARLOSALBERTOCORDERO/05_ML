#!/usr/bin/env python
import math

def main():
	int_a = [0]*9
	sum = 0
	mean = 0
	sd = 0
	for number in range(len(int_a)):
		inputNum = input ("Ingresar numero " + str(number) +": ")
		int_a[number] = int(inputNum)
		sum += int(inputNum)
	mean = sum/len(int_a)
	for number in range(len(int_a)):
		sd += ((int_a[number] - mean) ** 2)
	sd /= len(int_a)
	sd = math.sqrt(sd)
	print("mean: " + str(mean))
	print("sd: " + str(sd))

if __name__=='__main__':
	main()