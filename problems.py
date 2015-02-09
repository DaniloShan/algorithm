#! /usr/local/bin/python3
# ! /usr/bin/python3


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


r0 = TreeNode(5)
r0.left = TreeNode(4)
r0.left.left = TreeNode(11)
r0.left.left.left = TreeNode(7)
r0.left.left.right = TreeNode(2)
r0.right = TreeNode(8)
r0.right.left = TreeNode(13)
r0.right.right = TreeNode(4)
r0.right.right.right = TreeNode(1)

r1 = TreeNode(5)
r1.left = TreeNode(4)
r1.left.left = TreeNode(11)
r1.left.left.left = TreeNode(7)
r1.left.left.right = TreeNode(2)
r1.right = TreeNode(0)
r1.right.left = TreeNode(13)
r1.right.right = TreeNode(4)
r1.right.right.right = TreeNode(1)


class Solution(object):
    def __init__(self):
        self.min_so_far = 0
        self.max_so_far = 0

    def reverse_words(self, s):
        """Given an input string, reverse the string word by word.
        For example,
        Given s = "the sky is blue",
        return "blue is sky the".

        click to show clarification.

        Clarification:
        What constitutes a word?
        A sequence of non-space characters constitutes a word.
        Could the input string contain leading or trailing spaces?
        Yes. However, your reversed string should not contain leading or trailing spaces.
        How about multiple spaces between two words?
        Reduce them to a single space in the reversed string.
        """

        if type(s) is not str or not len(s.strip()):
            return ''
        s = s.split()
        s.reverse()
        return ' '.join(s)

    def is_palindrome(self, s):
        """Determine whether an integer is a palindrome. Do this without extra space.
        click to show spoilers.

        Some hints:
        Could negative integers be palindromes? (ie, -1)

        If you are thinking of converting the integer to string, note the restriction of using extra space.

        You could also try reversing an integer. However, if you have solved the problem "Reverse Integer",
        you know that the reversed integer might overflow. How would you handle such case?

        There is a more generic way of solving this problem.
        """

        if type(s) is not str:
            return False

        s = s.lower()
        result = ''

        for i in s:
            if i.isalpha() or i.isalnum():
                result += i

        if result[::-1] == result:
            return True
        else:
            return False

    def count_and_say(self, n):
        """The count-and-say sequence is the sequence of integers beginning as follows:
        1, 11, 21, 1211, 111221, ...

        1 is read off as "one 1" or 11.
        11 is read off as "two 1s" or 21.
        21 is read off as "one 2, then one 1" or 1211.
        Given an integer n, generate the nth sequence.

        Note: The sequence of integers will be represented as a string.
        """
        if not n > 0 and not type(n) is int:
            return "0"
        n -= 1
        count = 1
        temp = '1'
        result = ''

        prev_count = 1
        prev_num = 0

        while count <= n:
            count += 1
            for i in range(len(temp) + 1):
                if i < len(temp):
                    if prev_num:
                        if prev_num is temp[i]:
                            prev_count += 1
                        else:
                            result += str(prev_count) + str(prev_num)
                            prev_num = temp[i]
                            prev_count = 1
                    else:
                        prev_num = temp[i]
                elif temp[i - 1]:
                    result += str(prev_count) + str(temp[i - 1])

            temp = result

            result = ''
            prev_count = 1
            prev_num = 0

        print(temp)

        return str(temp)

    def reverse(self, x):
        """
        Reverse digits of an integer.

        Example1: x = 123, return 321
        Example2: x = -123, return -321

        click to show spoilers.

        Have you thought about this?
        Here are some good questions to ask before coding.
        Bonus points for you if you have already thought through this!

        If the integer's last digit is 0, what should the output be? ie, cases such as 10, 100.

        Did you notice that the reversed integer might overflow? Assume the input is a 32-bit integer,
        then the reverse of 1000000003 overflows. How should you handle such cases?

        Throw an exception? Good,
        but what if throwing an exception is not an option? You would then have to re-design
        the function (ie, add an extra parameter).
        """

        if type(x) is not int:
            return False

        return int(str(x)[::-1]) if x >= 0 else 0 - int(str(0 - x)[::-1])

    def atoi(self, s):
        """
        Implement atoi to convert a string to an integer.

        Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask
        yourself what are the possible input cases.

        Notes: It is intended for this problem to be specified vaguely (ie, no given input specs).
        You are responsible to
        gather all the input requirements up front.

        spoilers alert... click to show requirements for atoi.

        Requirements for atoi:
        The function first discards as many whitespace characters as necessary
        until the first non-whitespace character is found.
        Then, starting from this character, takes an optional initial plus or minus sign followed by as many
          numerical digits as possible, and interprets them as a numerical value.

        The string can contain additional characters after those that form the integral number, which are ignored and
        have no effect on the behavior of this function.

        If the first sequence of non-whitespace characters in str is not a valid integral number,
         or if no such sequence
        exists because either str is empty or it contains only whitespace characters, no conversion is performed.

        If no valid conversion could be performed, a zero value is returned.
        If the correct value is out of the range of
        representable values, INT_MAX (2147483647) or INT_MIN (-2147483648) is returned.
        """

        int_max = 2147483647
        int_min = -2147483648

        s = s.strip()

        func = int

        result = 0

        if not len(s):
            return 0

        if '.' in s:
            func = float

        try:
            s = [x if x in '+-0123456789' else False for x in s]
            s = s[:s.index(False) if False in s else len(s)]
            s = ''.join(s)
            result = func(s)
            if result > int_max:
                result = int_max
            if result < int_min:
                result = int_min
            return result

        except ValueError:
            return result

    def is_palindrome2(self, x):
        """
        # Palindrome Number

        determine whether an integer is a palindrome. Do this without extra space.

        click to show spoilers.

        Some hints:
        Could negative integers be palindromes? (ie, -1)

        If you are thinking of converting the integer to string, note the restriction of using extra space.

        You could also try reversing an integer. However, if you have solved the problem "Reverse Integer",
        you know that the reversed integer might overflow. How would you handle such case?

        There is a more generic way of solving this problem.
        """

        if x < 0 or type(x) != int:
            return False

        temp = 0
        n = x

        while n > 0:
            temp *= 10
            temp += n % 10
            n = int(n / 10)
        return True if temp == x else False

    def generate(self, num_rows):
        """
        # Pascal's Triangle

        Given numRows, generate the first numRows of Pascal's triangle.
        For example, given numRows = 5,
        Return

        [
             [1],
            [1,1],
           [1,2,1],
          [1,3,3,1],
         [1,4,6,4,1]
        ]

        """

        result, n = [[1]], 1

        if num_rows <= 0:
            return []

        while n < num_rows:
            result.append([1])
            for i, num in enumerate(result[n - 1]):
                result[n].append(num + result[n - 1][i + 1] if i < len(result[n - 1]) - 1 else 1)

            n += 1

        return result

    def get_row(self, row_index):
        """
        # Pascal's Triangle II
        Given an index k, return the kth row of the Pascal's triangle.

        For example, given k = 3,
        Return [1,3,3,1].

        Note:
        Could you optimize your algorithm to use only O(k) extra space?
        """

        result, n, pre_row = [1], 0, [[1]]

        if row_index < 0:
            return []

        while n < row_index:
            result = [1]

            for i, num in enumerate(pre_row):
                result.append(num + pre_row[i + 1] if i < len(pre_row) - 1 else 1)

            pre_row = result

            n += 1

        print(result)

    def max_product(self, num_list):
        """Maximum Product Subarray

        @param num_list, a list of integers
        @return an integer
        Find the contiguous subarray within an array (containing at least one number) which has the largest product.

        For example, given the array [2,3,-2,4],
        the contiguous subarray [2,3] has the largest product = 6.
        """

        if len(num_list) == 0:
            return 0
        if len(num_list) == 1:
            return num_list[0]

        max_temp, min_temp, result = num_list[0], num_list[0], num_list[0]

        for i, n in enumerate(num_list[1:]):
            pmax = max_temp * n
            pmin = min_temp * n

            max_temp = max(max(pmax, pmin), n)
            min_temp = min(min(pmax, pmin), n)
            result = max_temp if max_temp > result else result

        return result

    def roman_to_int(self, s):
        """Given a roman numeral, convert it to an integer.
        Input is guaranteed to be within the range from 1 to 3999.
        """
        roman_dic = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }

        if not len(s):
            return 0

        elif len(s) == 1:
            return roman_dic[s.upper()]

        pre_num = 0
        result = 0

        for num in s:
            num = roman_dic[num.upper()]

            if pre_num < num:
                result -= 2 * pre_num

            result += num
            pre_num = num

        return result

    def longest_common_prefix(self, s):
        """Write a function to find the longest common prefix string amongst an array of strings."""

        if not len(s) or not len(s[0]):
            return ''

        result = ''

        for letter in s[0]:
            result += letter

            for s in s:
                if not result in s or s.index(result) != 0:
                    return result[:-1]

        return result

    def has_path_sum(self, root, n):
        # @param root, a tree node
        # @param n, an integer
        # @return a boolean

        """Given a binary tree and a sum, determine if the tree has a root-to-leaf path
        such that adding up all the values along the path equals the given sum.

        For example:
        Given the below binary tree and sum = 22,
                  5
                 / \
                4   8
               /   / \
              11  13  4
             /  \      \
            7    2      1
        return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
        """

        if root is None:
            return False

        if root.left is None and root.right is None:
            return root.val == n

        return self.has_path_sum(root.left, n - root.val) or self.has_path_sum(root.right, n - root.val)

    """Minimum Depth of Binary Tree
    
    Given a binary tree, find its minimum depth.
    The minimum depth is the number of nodes along the
    shortest path from the root node down to the nearest leaf node.
    """

    def min_depth(self, root):
        if not root:
            return 0

        def get_depth(node, depth_this_node):
            if not node:
                return False

            if not node.left and not node.right:

                if depth_this_node < self.min_so_far or not self.min_so_far:
                    self.min_so_far = depth_this_node

                return False

            depth_this_node += 1

            return get_depth(node.left, depth_this_node) or get_depth(node.right, depth_this_node)

        get_depth(root, 1)

        return self.min_so_far or 0

    """Maximum Depth of Binary Tree
    Given a binary tree, find its maximum depth.
    The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
    """
    # @param root, a tree node
    # @return an integer
    def max_depth(self, root):

        if not root:
            return 0

        self.max_so_far = 0

        def get_depth(node, depth_this_node):
            if not node:
                return False

            if not node.left and not node.right:

                if depth_this_node > self.max_so_far or not self.max_so_far:
                    self.max_so_far = depth_this_node

                return False

            depth_this_node += 1

            return get_depth(node.left, depth_this_node) or get_depth(node.right, depth_this_node)

        get_depth(root, 1)

        return self.max_so_far or 0

    """Same Tree
    Given two binary trees, write a function to check if they are equal or not.
    Two binary trees are considered equal if they are structurally
    identical and the nodes have the same value.
    """
    # @param p, a tree node
    # @param q, a tree node
    # @return a boolean
    def is_same_tree(self, p, q):
        if not p or not q:
            return p == q

        if p.val == q.val:
            return self.is_same_tree(p.left, q.left) and self.is_same_tree(p.right, q.right)
        else:
            return False

    """Balanced Binary Tree 
    Given a binary tree, determine if it is height-balanced.
    For this problem, a height-balanced binary tree is defined as a
    binary tree in which the depth of the two subtrees of every
    node never differ by more than 1.
    """
    # @param root, a tree node
    # @return a boolean
    def is_balanced(self, root):

        def assert_balanced(node):
            if not node:
                return True

            if abs(depth(node.left) - depth(node.right)) > 1:
                return False

            return assert_balanced(node.left) and assert_balanced(node.right)


        def depth(node):
            if not node:
                return 0

            return 1 + max(depth(node.left), depth(node.right))

        return assert_balanced(root)

    """Majority Element
    Given an array of size n, find the majority element.
    The majority element is the element that appears more than n/2 times.
    You may assume that the array is non-empty and the majority element always exist in the array.
    """
    # @param num, a list of integers
    # @return an integer
    def majority_element(self, num):
        # hash table solution
        # majority, tmp_dic = 0, {}
        #
        # for n in num:
        #     tmp_dic[n] = tmp_dic[n] + 1 if n in tmp_dic else 1
        #     if majority not in tmp_dic:
        #         majority = n
        #     else:
        #         majority = n if tmp_dic[n] > tmp_dic[majority] else majority
        #
        # return majority if tmp_dic[majority] > len(num) / 2 else 0

        # one line solution
        # return num.sort() or num[len(num) // 2]

        # Moore voting algorithm
        majority = [False, 0]

        for n in num:
            if not majority[0]:
                majority[0] = n
                majority[1] = 1
            elif majority[0] == n:
                majority[1] += 1
            else:
                majority[1] -= 1
                if not majority[1]:
                    majority = [False, 0]

        return majority[0] or 0


    """Symmetric Tree
    Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

    For example, this binary tree is symmetric:

        1
       / \
      2   2
     / \ / \
    3  4 4  3
    But the following is not:
        1
       / \
      2   2
       \   \
       3    3
    Note:
    Bonus points if you could solve it both recursively and iteratively.

    confused what "{1,#,2,3}" means? > read more on how binary tree is serialized on OJ

    OJ's Binary Tree Serialization:
    The serialization of a binary tree follows a level order traversal,
    where '#' signifies a path terminator where no node exists below.

    Here's an example:
       1
      / \
     2   3
        /
       4
        \
         5
    The above binary tree is serialized as "{1,2,3,#,#,4,#,#,5}".
    """
    # @param root, a tree node
    # @return a boolean
    def is_symmetric(self, root):
        levels = {}

        def deep_first_search(node, levels, level_so_far):
            if not node:
                val = '#'
            else:
                val = node.val

            if not level_so_far in levels:
                levels[level_so_far] = []

            levels[level_so_far].append(val)

            level_so_far += 1

            return node and (node.left or node.right) and (deep_first_search(node.left, levels, level_so_far)
                                                           or deep_first_search(node.right, levels, level_so_far))

        deep_first_search(root, levels, 1)

        for key in levels:
            if len(levels[key]) % 2 != 0 and key != 1:
                return False
            if levels[key] != levels[key][::-1]:
                return False

        return True


    """Binary Tree Level Order Traversal
    Given a binary tree, return the level order traversal of its nodes' values.
     (ie, from left to right, level by level).

    For example:
    Given binary tree {3,9,20,#,#,15,7},
        3
       / \
      9  20
        /  \
       15   7
    return its level order traversal as:
    [
      [3],
      [9,20],
      [15,7]
    ]
    confused what "{1,#,2,3}" means? > read more on how binary tree is serialized on OJ.


    OJ's Binary Tree Serialization:
    The serialization of a binary tree follows a level order traversal,
    where '#' signifies a path terminator where no node exists below.

    Here's an example:
       1
      / \
     2   3
        /
       4
        \
         5
    The above binary tree is serialized as "{1,2,3,#,#,4,#,#,5}".
    """
    # @param root, a tree node
    # @return a list of lists of integers
    def level_order(self, root):
        if not root:
            return []
        levels = []

        def breadth_first_search(node, level_so_far):
            if not node:
                return False
            if level_so_far > len(levels):
                levels.append([])

            levels[level_so_far - 1].append(node.val)

            level_so_far += 1

            return breadth_first_search(node.left, level_so_far) or breadth_first_search(node.right, level_so_far)

        breadth_first_search(root, 1)

        return levels

    """Binary Tree Level Order Traversal II
     Given a binary tree, return the bottom-up level order traversal of its nodes' values.
      (ie, from left to right, level by level from leaf to root).

    For example:
    Given binary tree {3,9,20,#,#,15,7},
        3
       / \
      9  20
        /  \
       15   7
    return its bottom-up level order traversal as:
    [
      [15,7],
      [9,20],
      [3]
    ]
    confused what "{1,#,2,3}" means? > read more on how binary tree is serialized on OJ.


    OJ's Binary Tree Serialization:
    The serialization of a binary tree follows a level order traversal,
     where '#' signifies a path terminator where no node exists below.

    Here's an example:
       1
      / \
     2   3
        /
       4
        \
         5
    The above binary tree is serialized as "{1,2,3,#,#,4,#,#,5}".
    """
    # @param root, a tree node
    # @return a list of lists of integers
    def level_order_bottom(self, root):
        if not root:
            return []
        levels = []

        def breadth_first_search(node, level_so_far):
            if not node:
                return False
            if level_so_far > len(levels):
                levels.insert(0, [])

            levels[0 - level_so_far].append(node.val)

            level_so_far += 1

            return breadth_first_search(node.left, level_so_far) or breadth_first_search(node.right, level_so_far)

        breadth_first_search(root, 1)

        return levels

    """Plus One
    Given a non-negative number represented as an array of digits, plus one to the number.
    The digits are stored such that the most significant digit is at the head of the list.
    """
    # @param digits, a list of integer digits
    # @return a list of integer digits
    def plus_one(self, digits):
        result = []
        self.next = 0
        length = len(digits)
        if not length:
            return digits

        digits[-1] += 1

        while length:
            item = digits.pop()

            if self.next:
                item += 1
                self.next = 0

            if item < 10:
                result.insert(0, item)
            else:
                self.next = 1
                result.insert(0, item % 10)

            length -= 1
        else:
            if self.next:
                result.insert(0, 1)

        return result

    """Add Binary
    Given two binary strings, return their sum (also a binary string).

    For example,
    a = "11"
    b = "1"
    Return "100".
    """
    # @param a, a string
    # @param b, a string
    # @return a string
    def add_binary(self, a, b):
        # if not a or not b:
        #     return a or b
        #
        # return '{0:b}'.format(int(a, 2) + int(b, 2))
        if not a or not b:
            return a or b
        a, b = list(a), list(b)
        la, lb = len(a), len(b)
        length = la if la > lb else lb
        result = []
        self.next = 0

        while length:
            item = 0

            if self.next:
                item = 1
                self.next = 0

            item += (int(a.pop()) if len(a) else 0) + (int(b.pop()) if len(b) else 0)

            if item > 1:
                self.next = 1
                result.insert(0, str(item % 2))
            else:
                result.insert(0, str(item))

            length -= 1
        else:
            if self.next:
                result.insert(0, '1')

        return ''.join(result)

    """Factorial Trailing Zeroes
    Given an integer n, return the number of trailing zeroes in n!.

    Note: Your solution should be in logarithmic time complexity.

    Credits:
    Special thanks to @ts for adding this problem and creating all test cases.
    """
    # @return an integer
    def trailing_zeroes(self, n):
        if not n:
            return n

        self.result = 0

        while n >= 5:
            n //= 5
            self.result += n

        return self.result

    """Excel Sheet Column Title
    Given a positive integer, return its corresponding column title as appear in an Excel sheet.

    For example:

        1 -> A
        2 -> B
        3 -> C
        ...
        26 -> Z
        27 -> AA
        28 -> AB
    """
    # @return a string
    # TODO: damn it
    def convert_to_title(self, num):
        digits, n, results = [], 0, []
        title_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z']

        if num <= 26:
            return title_list[num - 1]

        while True:
            digit = 26 ** n

            if digit <= num:
                digits.append(digit)
                n += 1
            else:
                break

        for digit in digits[::-1]:
            result = num // digit
            if num % digit == 0 and digit != 1:
                result -= 1
            results.append(result)
            num -= result * digit

        print('==========')
        print(digits)
        print(num)
        print(results)

        result = ''

        for n in results:
            result += title_list[n - 1]

        if num:
            result += title_list[num - 1]

        return result

    """Valid Parentheses
    Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
     determine if the input string is valid.

    The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
    """
    # @return a boolean
    def is_valid(self, s):
        tmp = []
        match = '()[]{}'

        for item in s:
            if len(tmp) and tmp[-1] + item in match:
                tmp.pop()
            else:
                tmp.append(item)

        return len(tmp) == 0

    """
    # another solution
        pointer, l = 0, len(s)
        match, tmp = '(){}[]', list(s)

        while(l > 0):
            if pointer and tmp[pointer - 1] + tmp[pointer] in match:
                tmp.pop(pointer)
                tmp.pop(pointer - 1)
                pointer -= 1
            else:
                pointer += 1

            l -= 1

        return len(tmp) == 0
    """

    """Merge Sorted Array
    Given two sorted integer arrays A and B, merge B into A as one sorted array.

    Note:
    You may assume that A has enough space (size that is greater or equal to m + n)
    to hold additional elements from B.
    The number of elements initialized in A and B are m and n respectively.
    """
    # TODO: unknown error
    # @param A  a list of integers
    # @param m  an integer, length of A
    # @param B  a list of integers
    # @param n  an integer, length of B
    # @return nothing
    def merge(self, A, m, B, n):
        i, j = 0, 0

        if m == 0:
            while j < n:
                A.append(B[j])
                j += 1
            return

        while i < m and j < n:
            if not B[j]:
                break
            elif A[i] > B[j]:
                A.insert(i, B[j])
                j += 1

            i += 1

        if j < n:
            while j < n:
                if A[-1] > B[j]:
                    A.insert(-1, B[j])
                else:
                    A.append(B[j])

                j += 1


    """Compare Version Numbers
    Compare two version numbers version1 and version2.
    If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.

    You may assume that the version strings are non-empty and contain only digits and the . character.
    The . character does not represent a decimal point and is used to separate number sequences.
    For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.

    Here is an example of version numbers ordering:

    0.1 < 1.1 < 1.2 < 13.37
    """
    # @param version1, a string
    # @param version2, a string
    # @return an integer
    def compare_version(self, version1, version2):
        arr_1, arr_2 = version1.split('.'), version2.split('.')
        points = 0

        if not len(arr_1) and len(arr_2):
            return -1
        elif not len(arr_2) and len(arr_1):
            return 1

        if len(arr_1) > len(arr_2):
            larger, smaller = arr_1, arr_2
        else:
            smaller, larger = arr_1, arr_2

        for i, n in enumerate(larger):
            if i >= len(smaller):

                if points == 0 and int(n) > 0:
                    return 1 if larger == arr_1 else -1
                else:
                    if points == 1 and larger == arr_1:
                        return 1
                    if points == -1 and larger == arr_2:
                        return -1
                continue
            if not points:
                if int(arr_1[i]) > int(arr_2[i]):
                    return 1
                elif int(arr_1[i]) < int(arr_2[i]):
                    return -1

        return points

"""Min Stack
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
"""
class MinStack:
    def __init__(self):
        self.min_stack = []
        self.stack = []

    # @param x, an integer
    # @return an integer
    def push(self, x):
        self.stack.append(x)
        if len(self.min_stack) == 0 or x <= self.min_stack[-1]:
            self.min_stack.append(x)
        return x

    # @return nothing
    def pop(self):
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    # @return an integer
    def top(self):
        return self.stack[-1]

    # @return an integer
    def getMin(self):
        return self.min_stack[-1]


solution = Solution()

# print(s.minDepth(r0))
# print(s.minDepth({}))
# print(r0.right.val)
# print(r1.right.val)
# print(solution.isSameTree(r0, r1))
# r3 = TreeNode(1)
# r3.right = TreeNode(2)
# r3.left = TreeNode(2)
# # r3.right.right = TreeNode(3)
# r3.right.left = TreeNode(4)
# r3.left.right = TreeNode(4)
# # r3.left.left = TreeNode(3)

# r4 = TreeNode(2)
# r4.left = TreeNode(3)
# r4.right = TreeNode(3)
#
# r4.left.left = TreeNode(4)
# r4.left.right = TreeNode(5)
# r4.right.left = TreeNode(5)
# r4.right.right = TreeNode(4)
#
# r4.left.left.left = TreeNode('#')
# r4.left.left.right = TreeNode('#')
# r4.left.right.left = TreeNode(8)
# r4.left.right.right = TreeNode(9)
# r4.right.left.left = TreeNode('#')
# r4.right.left.right = TreeNode('#')
# r4.right.right.left = TreeNode(9)
# r4.right.right.right = TreeNode(8)
#
# r5 = TreeNode(1)
# r5.left = TreeNode(2)
# r5.right = TreeNode(3)
# r5.left.left = TreeNode(4)
# r5.left.right = TreeNode(5)
#
# print(solution.level_order_bottom(r0))
# print(solution.level_order_bottom(r1))
# print(solution.level_order_bottom(r3))
# print(solution.level_order_bottom(TreeNode(1)))
# print(solution.level_order_bottom(r4))
# print(solution.level_order_bottom(r5))

# print(solution.plus_one([1,2,3,4,5,6]))
# print(solution.plus_one([9]))
# print(solution.plus_one([6,5,3,2,9]))
# print(solution.plus_one([4,3,5,9,2,8,9,9,9]))

# print(solution.trailing_zeroes(21))
# print(solution.trailing_zeroes(5))
# print(solution.trailing_zeroes(30))

# print(solution.convert_to_title(12345))
# print(solution.convert_to_title(52))
# print(solution.convert_to_title(78))
# print(solution.convert_to_title(27))
# print(solution.convert_to_title(701))

# arr1 = []
# arr2 = [1]
# print(solution.merge(arr1, len(arr1), arr2, len(arr2)))
# print(arr1)

print(solution.compare_version('1', '1.1'))