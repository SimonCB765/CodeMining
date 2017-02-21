"""Process a journal table into a format that is quicker to read."""

# Python imports.
from collections import defaultdict
import logging

# User imports.
from . import parse_patient_entry

# Globals.
LOGGER = logging.getLogger(__name__)


def main(fileJournalTable, fileProcessedData):
    """Process a journal table and convert it to a more standardised TSV format.

    :param fileJournalTable:    The location of the file containing the SQL dump file with the patient data.
    :type fileJournalTable:     str
    :param fileProcessedData:   The location to save the processed journal table data.
    :type fileProcessedData:    str

    """

    LOGGER.info("Starting journal table pre-processing.")

    # Convert the journal table into a standard format, ignoring any entries that are missing either a patient ID or
    # a code.
    currentPatient = None  # The ID of the patient who's record is currently being built.
    patientHistory = defaultdict(int)  # The data for the current patient.
    uniqueCodes = set()  # The codes used in the dataset.

    with open(fileJournalTable, 'r') as fidJournalTable, open(fileProcessedData, 'w') as fidProcessed:
        # Process journal table.
        for line in fidJournalTable:
            if line.startswith("insert"):
                # The line contains information about a row in the journal table.
                entries = parse_patient_entry.main(line)
                patientID = entries[0]
                code = entries[1]

                if patientID and code:
                    # The entry is valid as it has both a patient ID and code recorded for it.
                    uniqueCodes.add(code)

                    if patientID != currentPatient and currentPatient:
                        # A new patient has been found and this is not the first line of the file, so record the old
                        # patient and reset the patient data for the new patient.

                        # Write out the patient's history sorted by date from oldest to newest.
                        fidProcessed.write("{:s}\t{:s}\n".format(
                            patientID, '\t'.join(["{:s}:{:d}".format(i, patientHistory[i]) for i in patientHistory])
                        ))

                        # Reset the history and record of codes the patient has to prepare for the next patient.
                        patientHistory.clear()
                    currentPatient = patientID

                    # Add the entry to the patient's history.
                    patientHistory[code] += 1

    # Write out the codes in the dataset.
    uniqueCodes = sorted(uniqueCodes)
    with open(fileProcessedData, 'r+') as fidProcessed:
        processed = fidProcessed.read()
        fidProcessed.seek(0)
        fidProcessed.write("{:s}\n{:s}".format('\t'.join(uniqueCodes), processed))
