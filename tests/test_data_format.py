"""Regression checks that do not load models or rewrite saved datasets."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_script('convert_dataset')
checker = load_script('check_format')
CODE = '#include <iostream>\nint main() { return 0; }'


def example(answer=CODE, statement='题目描述\n求和\n样例\n1\n说明\n1 <= n <= 100'):
    return {'conversations': [{'from': 'human', 'value': statement}],
            'chosen': {'value': answer}}


class DatasetFormatTests(unittest.TestCase):
    def test_language_free_fence_does_not_produce_empty_answer(self):
        row = converter.convert(example(f'```\n{CODE}\n```\n解释'))
        self.assertTrue(row['valid'])
        self.assertEqual(checker.extract_completion(row), CODE)

    def test_cpp_fence_and_plain_code(self):
        for answer in (CODE, f'```cpp\n{CODE}\n```', f'```c++\r\n{CODE}\r\n```'):
            with self.subTest(answer=answer):
                row = converter.convert(example(answer))
                self.assertTrue(row['valid'])
                self.assertEqual(checker.extract_completion(row).replace('\r\n', '\n'), CODE)

    def test_statement_keeps_constraints_after_samples_and_title(self):
        statement = '# 标题\n题目描述\n求和\n样例\n1\n说明\n1 <= n <= 100'
        self.assertIn(statement, converter.convert(example(statement=statement))['text'])

    def test_statement_without_sample_is_preserved(self):
        self.assertTrue(converter.convert(example(statement='题目描述\n约束：n < 100'))['valid'])

    def test_missing_statement_and_answer_are_invalid(self):
        for row in ({}, example(answer='```\n```'), example(statement='没有题面'),
                    {'conversations': None}, {'conversations': [{'value': None}]},
                    example(answer=None)):
            with self.subTest(row=row):
                self.assertEqual(converter.convert(row), {'text': '', 'valid': False})

    def test_checker_reads_assistant_not_user_code(self):
        row = converter.convert(example(statement='题目描述\n```cpp\nUSER CODE\n```'))
        self.assertEqual(checker.extract_completion(row), CODE)

    def test_checker_rejects_missing_text_or_end_marker(self):
        for row in ({'completion': CODE}, {'text': None},
                    {'text': checker.ASSISTANT_MARKER + CODE},
                    {'text': checker.ASSISTANT_MARKER + CODE + '<|im_end|>garbage'}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                checker.extract_completion(row)

    def test_checker_checks_empty_answer_beyond_preview(self):
        class FakeDataset(list):
            column_names = ['text']
        rows = FakeDataset([converter.convert(example()) for _ in range(10)] +
                           [{'text': checker.ASSISTANT_MARKER + ' <|im_end|>'}])
        fake_datasets = types.SimpleNamespace(load_from_disk=lambda path: {'train': rows})
        with patch.dict(sys.modules, {'datasets': fake_datasets}), patch.object(sys, 'argv', ['check_format.py']):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(checker.main(), 1)
            self.assertIn('10: INVALID: empty assistant answer', output.getvalue())

    def test_import_has_no_datasets_dependency_or_io(self):
        with patch.dict(sys.modules, {'datasets': None}):
            load_script('convert_dataset')
            load_script('check_format')


if __name__ == '__main__':
    unittest.main()
