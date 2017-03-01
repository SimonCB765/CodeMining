"""This module contains a class, Configuration, that holds the configuration parameters of the running program."""

# Python import.
import json


class Configuration(object):

    def __init__(self, **kwargs):
        """Initialise a Configuration object.

        :param kwargs:  Keyword arguments to initialise.
        :type kwargs:   dict

        """

        self._configParams = {}
        self.set_from_dict(kwargs)

    def get_param(self, path):
        """Extract a configuration parameter from the dictionary of parameters.

        :param path:    The path through the configuration parameter dictionary to use to extract the parameter.
        :type path:     list[str]
        :return:        The parameter's value or an indication that the parameter does not exist.
        :rtype:         bool, list | int | str | float | dict | None
                            1st element indicates whether the parameter was found.
                            2nd element is the parameter's value if found and the name of the first missing dictionary
                                key if not found (e.g. "A" if self._configParams["B"]["A"]["C"] fails because "A"
                                is not a key in the self._configParams["B"] dictionary.

        """

        paramFound = False  # Whether the parameter was found.
        paramValue = self._configParams  # The value to return.
        for i in path:
            if i in paramValue:
                # The next element in the path was found, so keep looking.
                paramValue = paramValue[i]
            else:
                # The parameter was not found, so terminate the search.
                paramValue = i
                break
        else:
            # The parameter was found as all elements in the path were found.
            paramFound = True

        return paramFound, paramValue

    def set_from_dict(self, paramsToAdd):
        """Set configuration parameters from a dictionary of parameters.

        This will overwrite any existing parameters with the same name.

        :param paramsToAdd:  Parameters to add.
        :type paramsToAdd:   dict

        """

        self._configParams.update(paramsToAdd)

    def set_from_json(self, config):
        """Add parameters to a Configuration object from a JSON formatted file.

        Any configuration parameters that the user has defined will overwrite existing parameters with the same name.
        Storing defaults will never overwrite user-defined or pre-existing parameters.

        :param config:          The location of a JSON file containing the configuration information to add.
        :type config:           str

        """

        # Extract the JSON data.
        fid = open(config, 'r')
        config = json.load(fid)
        self.set_from_dict(config)
        fid.close()
