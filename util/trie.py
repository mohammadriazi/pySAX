# Implementation of Trie

class Node(object):

	def __init__(self):
		self.children = {}
		self.isword = False

	def isleaf(self):
		return True if not self.children else False


class Trie(object):

	def __init__(self):
		self.root = None


	def insert(self,word):

		if not self.root:
			self.root = Node()
		current_node = self.root

		for c in list(word):
			if c not in current_node.children:
				current_node.children[c] = Node()
			current_node = current_node.children[c]
		current_node.isword = True


	def print_words(self,root,path=""):
		# DFS implementation of Trie traversal
		if root.isword:
			print(path)
		if root.isleaf():
			return
		for key,child in root.children.items():
			self.print_words(child,path+str(key))


	def has_word(self,word):

		current_node = self.root
		trav_path = ""
		for c in list(word):
			if c not in current_node.children:
				return False
			trav_path += str(c)
			current_node = current_node.children[c]
			if current_node.isword and trav_path==word:
				return True
			if current_node.isleaf():
				return False


	def get_first_word_in_item(self,item):
		current_node = self.root
		trav_path = ""
		for c in list(item):
			if c not in current_node.children:
				return None
			trav_path += str(c)
			current_node = current_node.children[c]
			if current_node.isword:
				return trav_path
			if current_node.isleaf():
				return None


	def get_root(self):
		return self.root
