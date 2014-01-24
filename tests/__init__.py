import __main__
defaultGlobalDict = __main__.__dict__
import os, unittest
from types import StringTypes
    
# shared, default config root for testing
_root = None

def importSuite(string_or_list, globalDict=defaultGlobalDict):
    """
    @description

      Imports and returns the test suite, as specified by 'suite_specifiation'.
    
      Example values for 'string_or_list':
      1. 'regressor.test.unittest_suite'
      2. ['regressor.test.unittest_suite', 'regressor.test.doctest_suite']
      3. [callable_function1, callable_function2, ...]

    @arguments

      string_or_list -- string, or list of string, or list of object

    @return

      test_suite -- TestSuite object

    @exceptions

    @notes
    """
    from unittest import TestSuite
    test_funcs = importSequence(string_or_list, globalDict)
    return TestSuite([test_func() for test_func in test_funcs])

def importObject(string, globalDict=defaultGlobalDict):
    """
    @description

      Imports the object specified by 'string' using 'importString()'
        
      (If 'string' is not a string, then it considers it already imported and merely returns
       the input as-is.)

    @arguments

      string -- string or object
      
    @return

      imported_object -- object -- what importString(string) returns

    @exceptions

    @notes
    
    """
    if isinstance(string, StringTypes):
        return importString(string, globalDict)

    else:
        object = string
        return object

def importSequence(string_or_list, globalDict=defaultGlobalDict):
    """
    @description

      Import the items specified by 'string_or_list' and return as objects.

      If 'string_or_list' is a string or unicode object, treat it as a
      comma-separated list of import specifications, and return a
      list of the imported objects.

      If the result is not a string but is iterable, return a list
      with any string/unicode items replaced with their corresponding
      imports.
    
      Example values for 'string_or_list':
      1. 'regressor.test.unittest_suite'
      2. ['regressor.test.unittest_suite', 'regressor.test.doctest_suite']
      3. [callable_function1, callable_function2, ...]

    @arguments

      string_or_list -- string, or list of string, or list of object

    @return

      imported_objects -- list of object -- where each object is a callable func

    @exceptions

    @notes
    
    """
    if isinstance(string_or_list, StringTypes):
        string = string_or_list
        strings = string.split(',')
        return [importString(string.strip(), globalDict) for string in strings]
    else:
        _list = string_or_list
        return [importObject(object, globalDict) for object in _list]


def importString(name, globalDict=defaultGlobalDict):
    """
    @description

      Import an item specified by 'name' (a string)

      Example Usage::

        attribute1 = importString('some.module:attribute1')
        attribute2 = importString('other.module:nested.attribute2')

      'importString' imports an object from a module, according to an
      import specification string: a dot-delimited path to an object
      in the Python package namespace.  For example, the string
      '"some.module.attribute"' is equivalent to the result of
      'from some.module import attribute'.

    @arguments

      name -- string --
      
    @return

      imported_object -- object

    @exceptions

    @notes

      For readability of import strings, it's sometimes helpful to use a ':' to
      separate a module name from items it contains.  It's optional, though,
      as 'importString' will convert the ':' to a '.' internally anyway.
    
    """
    if ':' in name:
        name = name.replace(':','.')

    path  = []

    for part in filter(None,name.split('.')):

        if path:

            try:
                item = getattr(item, part)
                path.append(part)
                continue

            except AttributeError:
                pass

        path.append(part)
        item = __import__('.'.join(path), globalDict, globalDict, ['__name__'])

    return item

    

def addModules(dirname):
    """
    @description

    @arguments
      
    @return

    @exceptions

    @notes
    """
    mod_names = []

    for filename in os.listdir(dirname):
        pathname = dirname + '/' + filename
        if (pathname[-8:] == '_test.py'):
            mod_names.append(os.path.splitext(os.path.split(pathname)[1])[0])
        elif ((os.path.isdir(pathname)) and
              (not os.path.islink(pathname)) and
              (filename[0] != '.')):
            mod_names.extend(addModules(pathname))

    return mod_names
    
def default_test_suite(module=None):
    """
    @description

    @arguments
      
    @return

    @exceptions

    @notes
    """
    dir = os.path.dirname(getattr(module,'__file__',''))
    modules = map(__import__, addModules(dir))
    load = unittest.defaultTestLoader.loadTestsFromModule
    return unittest.TestSuite(map(load, modules))

allSuites = [
    'adts.test.test_suite',
    'engine.test.test_suite',
    'problems.test.test_suite',
    'regressor.test.test_suite',
    'util.test.test_suite',
]

def test_suite():
    return importSuite(allSuites, globals())

if __name__=="__main__":
    unittest.main(defaultTest='test_suite') 
