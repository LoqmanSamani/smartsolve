import unittest

from unittest import TestCase


def sum(a, b):
    return a + b  





class SumTest(TestCase):


    # arrange one case for all functions in the class
    def setUp(self):
        self.a = 20
        self.b = 10

    def test_func_2(self):

        result = sum(self.a, self.b)

        self.assertEqual(result, self.a + self.b)



    def test_sum(self):
        

        # Act
        result = sum(self.a, self.b)

        # Assert

        self.assertEqual(result, self.a + self.b)



class LearnAssertion(TestCase):


    def setUp(self):

        self.a = 12
        self.b = 12



    def test_func(self):

        if self.a == self.b:

            print("They are the same!")
        
        else:
            print("They are not the same")

    # how to test equality ???

    def test_equality(self):

        self.assertEqual(self.a, self.b)


    def test_equality(self):
         a = 12
         b = 10

         self.assertNotEqual(a, b)



    def test_equality(self):

        a = 90
        b = 90

        # self.assertEqual(a, b, msg=' 89 is not equal 90!')

        self.assertIs(a, b)
            # assetIsNot ??
            # assertIsInstance ??
            # assertNotIsInstance ??
            # assertIsNone ??
            # assertIsNotNone ??
            # assertTrue ??
            # assertFalse ??
           



def throw_ex(var):
    
    if var == 100:

        raise Exception("NOT A VALID NUMBER!")
    
    else:

        return True
    

class LearnUnitTest(TestCase):

    def test_sample(self):

        self.assertEqual(throw_ex(100),  True)


# the same can be done using assertRaises !

    
class LearnUnitTest(TestCase):

    def test_sample(self):

        self.assertRaises(Exception, throw_ex, 100)



    


    




    






if __name__ == "__main__":
    unittest.main()



