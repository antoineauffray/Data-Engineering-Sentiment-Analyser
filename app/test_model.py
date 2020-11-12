import unittest
from unittest.mock import patch

import model


class FlaskTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch('model.predict')
    def test_positive(self, mock_get_model):
        mock_get_model.return_value = 'positive'
        result = model.svm_model('I love white color')
        self.assertEqual(result, 'positive')

    @patch('model.predict')
    def test_neutral(self, mock_get_model):
        mock_get_model.return_value = 'neutral'
        result = model.svm_model('I white')
        self.assertEqual(result, 'neutral')

    @patch('model.predict')
    def test_negative(self, mock_get_model):
        mock_get_model.return_value = 'negative'
        result = model.svm_model('I hate white color')
        self.assertEqual(result, 'negative')

if __name__ == '__main__':
    unittest.main()
