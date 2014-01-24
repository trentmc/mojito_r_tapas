"""Schema.py

Holds knowledge about allowable topology combinations.

"""
from itertools import izip
import types

from util import mathutil

class Schema(dict):
    """
    @description
      
    @attributes

      self -- dict mapping varname : possible_values_list
      
    @notes
    """
    def __init__(self, *args):
        #initialize parent class
        dict.__init__(self,*args)
        for varname, varval in self.items():
            assert isinstance(varname, types.StringType)
            assert isinstance(varval, types.ListType)
        self.checkConsistency()

    def checkConsistency(self):
        """Will raise a ValueError if not consistent.
        Checks include:
        -are there duplicates in any schema's values?
        """
        for key, val in self.items():
            if len(val) != len(set(val)):
                raise ValueError(self)
                
    def __str__(self):
        """Override str()"""
        all_vars = self.keys()
        return self.str2(all_vars)
    
    def str2(self, varnames):
        """Output self, though just at the varnames given"""
        varnames = sorted(varnames)
        s = "{"
        for i, varname in enumerate(varnames):
            s += "%s: %s" % (varname, self[varname])
            if i < len(varnames)-1: s += ", "
        s += "}"
        return s

class Schemas(list):
    """
    @description
      
    @attributes

      self -- list of schema
      
    @notes
    """
    
    def __init__(self, *args):
        #initialize parent class
        list.__init__(self,*args)

        #consistent?
        self.checkConsistency()

    def checkConsistency(self):
        """Will raise a ValueError if not consistent.
        Currently, checks include:
        -call to each schema's _checkConsistency
        -FIXME: are there any overlaps in logic input space?
        -
        """
        for schema in self:
            assert isinstance(schema, Schema)
            schema.checkConsistency()

    def __str__(self):
        """Override str()"""
        s = "[\n"
        for schema in self:
            s += str(schema) + ",\n"
        s += "]\n"
        return s

    def compactStr(self):
        """Like str(), but only gives the variables with shared values once,
        and therefore it is simpler to look at.
        """
        #corner case...
        if len(self) <= 1:
            return str(self)
        
        #main case
        shared_vars, shared_schema = [], Schema()
        ref_schema = self[0]
        for var in ref_schema.keys():
            var_is_shared = True
            for schema in self[1:]:
                if not schema.has_key(var):
                    var_is_shared = False
                    break
                if schema[var] != ref_schema[var]:
                    var_is_shared = False
                    break
            if var_is_shared:
                shared_vars.append(var)
                shared_schema[var] = schema[var]

        s = "\n"
        s += "Schema values shared across all schemas: %s\n" % shared_schema
        s += "Non-shared values, per schema:\n"
        s += "[\n"
        for schema in self:
            diff_varnames = mathutil.listDiff(schema.keys(), shared_vars)
            s += schema.str2(diff_varnames) + ",\n"
        s += "]\n"
        return s

    def numPermutations(self):
        """
        @description

          Returns # possible topology permutations

        @arguments

          <<none>> (gets info from self)

        @return

          count -- int

        @exceptions

        @notes
        """
        count = 0
        for schema in self:
            assert isinstance(schema, Schema), schema
            schema_count = 1
            for varname, possible_values in schema.items():
                assert isinstance(possible_values, types.ListType), \
                       possible_values
                schema_count *= len(possible_values)
            count += schema_count
            
        return count

    def coversSpace(self):
        """If the 'space' is defined by the possible variables and values that this Schemas()
        has, then this routine returns True if this Schemas covers off all variables.
        """
        #determine the space
        max_val_per_var = self._possibleValuesSpace()

        #check if the space is covered
        ordered_vars = sorted(max_val_per_var.keys())            # var_i : varname
        bases = [max_val_per_var[var]+1 for var in ordered_vars] #var_i : max_val+1
        all_variable_combinations = mathutil.permutations(bases)
        space_is_covered = True
        for variable_combination in all_variable_combinations:
            found_a_covering_schema = False
            for schema in self:
                if self._schemaCoversValues(schema, ordered_vars, variable_combination):
                    found_a_covering_schema = True
                    break
            if not found_a_covering_schema:
                space_is_covered = False
                break
            
        return space_is_covered

    def _possibleValuesSpace(self):
        """
        Let 'space' be defined by the possible variables and values that this Schemas() has.

        Some rules:
        -a Schema covers a variable if that variable is not noted at all
        -the range of values for a variable are defined by 0, 1, ..., max value encountered

        Returns a dict of variable_name : maximum_value
        """
        max_val_per_var = {} # dict of varname : maximum_value
        for schema in self:
            for (var, values) in schema.iteritems():
                if not max_val_per_var.has_key(var):
                    max_val_per_var[var] = max(values)
                else:
                    max_val_per_var[var] = max(max_val_per_var[var], max(values))

        return max_val_per_var

    def _schemaCoversValues(self, schema, ordered_vars, variable_combination):
        """Returns True if 'schema' covers the target 'variable_combination'.
        'variable_combination' is a list of values, where variable_combination[i]
        corresponds to ordered_vars[i].
        """
        for (var, target_value) in izip(ordered_vars, variable_combination):
            if schema.has_key(var) and (target_value not in schema[var]):
                return False
        return True

    def merge(self):
        """
        @description

          Tries to find schemas where the possible_values lists
          can be merged, and merges them.

          Example: if the input is:
            [{'loadrail_is_vdd': [0, 1], 'cascode_recurse': [1]}
            {'loadrail_is_vdd': [0, 1], 'cascode_recurse': [0]}]
          Then it output is:
            [{'loadrail_is_vdd': [0, 1], 'cascode_recurse': [0,1]}]

        @arguments

          <<none>> (gets list of schemas info from self)

        @return

          <<may modify self to shrink list size>>

        @exceptions

        @notes
        """
        #first, some variables only have a maximum value of 0.  Remove them before proceeding
        max_val_per_var = self._possibleValuesSpace()
        vars_to_delete = [var for (var, max_val) in max_val_per_var.iteritems()
                          if max_val == 0]
        for schema in self:
            for var in vars_to_delete:
                if schema.has_key(var):
                    del schema[var]    
        
        #do the main work
        while True:
            found_merge = self._mergeOnceIfPossible()
            if not found_merge:
                break

        self.checkConsistency()
        return

    def _mergeOnceIfPossible(self):
        """
        @description

          Tries to find _any_ merge, and performs it if possible.

        @arguments

          <<none>> (gets info from self)

        @return

          found_merge -- bool -- found a merge to do?
          <<may modify self>>

        @exceptions

        @notes
        """
        #corner case: self is empty
        if len(self) == 0:
            return False

        #typical case
        for i, schema_i in enumerate(self):
            for j, schema_j in enumerate(self):
                if j <= i: continue

                vars_i = sorted(schema_i.keys())
                vars_j = sorted(schema_j.keys())
                if vars_i != vars_j: continue #no match: vars not identical

                values_i = [schema_i[var] for var in vars_i]
                values_j = [schema_j[var] for var in vars_j]
                diff_locs = [loc
                             for loc, (value_i, value_j) in
                             enumerate(zip(values_i, values_j))
                             if value_i != value_j]
                num_diff = len(diff_locs)
                num_vars = len(schema_i)

                ##tlm - the following line is commented out, because
                # it can happen after a novelty operation.  Instead
                # we can merge by merely keeping all but one schema.
                #assert num_diff > 0, "should never be identical"
                if num_diff == 0:
                    new_schemas = [schema for (k, schema) in enumerate(self)
                                   if k != j]
                    list.__init__(self, new_schemas)
                    return True
                    
                if num_diff > 1: continue #no match: too many different values

                #match found!
                merge_var = vars_i[diff_locs[0]]
                merge_val = values_i[diff_locs[0]] + values_j[diff_locs[0]]
                merge_val = sorted(list(set(merge_val)))

                new_schemas = []
                for k,schema in enumerate(self):
                    if k == i:
                        merged_schema = schema
                        merged_schema[merge_var] = merge_val
                        new_schemas.append(schema)
                    elif k == j:
                        pass
                    else:
                        new_schemas.append(schema)

                list.__init__(self, new_schemas)
                return True

        return False


def combineSchemasList(list_of_schemas):
    """Create a new Schemas which is the combinatorial explosion of the incoming list of
    schemas.
    Typical application is for a CompoundPart, where each 'Schemas' entry in list_of_schemas
    comes from an embedded part, and we want to create a new schema.
    """
    #preconditions
    assert isinstance(list_of_schemas, types.ListType)
    assert len(list_of_schemas) >= 1
    for schemas in list_of_schemas:
        assert isinstance(schemas, Schemas)
    
    #corner case
    if len(list_of_schemas) == 1:
        assert list_of_schemas[0].coversSpace()
        return list_of_schemas[0]

    #base case
    elif len(list_of_schemas) == 2:
        return combineTwoSchemas(list_of_schemas[0], list_of_schemas[1])

    #recursive case
    else:
        combined01 = combineTwoSchemas(list_of_schemas[0], list_of_schemas[1])
        return combineSchemasList([combined01] + list_of_schemas[2:])

def combineTwoSchemas(schemas1, schemas2):
    """Create a new Schemas which is the combinatorial explosion of the _two_ incominb Schemas objects.
    Helper for combineSchemasList().
    """
    #preconditions
    assert isinstance(schemas1, Schemas)
    assert isinstance(schemas2, Schemas)
    assert schemas1.coversSpace()
    assert schemas2.coversSpace()

    #main work
    new_schemas = Schemas()
    for schema1 in schemas1:
        for schema2 in schemas2:
            assert isinstance(schema1, Schema) and isinstance(schema2, Schema)
            new_schema = Schema()
            for var in set(schema1.keys() + schema2.keys()):
                values = set([])
                if schema1.has_key(var): values |= set(schema1[var])
                if schema2.has_key(var): values |= set(schema2[var])
                new_schema[var] = sorted(values)
            new_schemas.append(new_schema)

    new_schemas.merge()

    return new_schemas
    
