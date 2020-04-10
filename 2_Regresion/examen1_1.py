#!/usr/bin/env python

def main():
	good = False
	letters = set()

	while not good:
		string = input ("Ingresar palabra: ")
		size = len(string)

		if 3 <= size <= 9:
			for char in string:
				letters.add(char)

			if (2 == len(letters)):
				for char in letters:
					if string.count(char) == 1:
						print("Letter", char, "at index", string.index(char))
						good = True



if __name__=='__main__':
	main()