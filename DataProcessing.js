svgWidth = 1200;  // The width of the SVG element containing the diagram.
svgHeight = 800;  // The height of the SVG element containing the diagram.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth)
        .attr("height", svgHeight);

// Define the arrowhead shape.
var arrowheadWidth = 6;
var arrowheadHeight = 4;
var definitions = svg.append("defs");
var arrowhead = definitions.append("marker")
    .attr("id", "arrowhead")
    .attr("refX", 0)  // Set refX so that the left edge of the marker is affixed to the end of the arrow's line.
    .attr("refY", arrowheadHeight / 2)  // Set refY so that the middle of the marker (vertically) is affixed to the end of the arrow's line.
    .attr("markerWidth", arrowheadWidth)
    .attr("markerHeight", arrowheadHeight)
    .attr("orient", "auto");
arrowhead.append("path")
    .attr("d", "M0,0V" + arrowheadHeight + "L" + arrowheadWidth + "," + (arrowheadHeight / 2) + "Z");

// Read the data in.
var dataAccessorFunction = function(d)
    {
        return { ID : parseInt(d.PatientID), Codes : (d.Codes).split(",")};
    }
d3.tsv("/Data/DataProcessing.txt", dataAccessorFunction, function(error, data)
    {
        // Sort the data by patient ID.
        data = data.sort(function(a, b) { return d3.ascending(a.ID, b.ID); });

        // Create the raw data that will be displayed.
        var inputDataRows = 10;
        var rawData = generate_raw_data(data, inputDataRows);
        rawData = rawData.sort(function(a, b) { return d3.ascending(a[0], b[0]); });
        rawData.push(["...", "..."]);

        // Process the data.
        var processedNumberCols = 5;
        var processedNumberRows = 10;
        var processedData = process_data(data);
        var processedDataToDisplay = generate_process_display_data(processedData, processedNumberRows, processedNumberCols);

        // Create the final subset.
        var finalNumberCols = 5;
        var finalNumberRows = 5;
        var finalData = subset_data(processedData, finalNumberRows, finalNumberCols)

        // Define the widths, heights and number of header rows for the tables cells.
        // If there are fewer widths than columns, then the widths will be padded using the last value in the widths array.
        var headerRows = 2;
        var cellWidths = [75, 50];
        var cellHeight = 25;

        // Define table positioning variables.
        var tableVerticalLocation = 50;
        var interTableSpacing = 100;  // Spacing between tables.

        // Create the input data table.
        var inputDataCoords = { x : 40, y : tableVerticalLocation };
        var inputDataTable = svg.append("g")
            .classed("table", true)
            .attr("transform", "translate(" + inputDataCoords.x + "," + inputDataCoords.y + ")");
        create_table(inputDataTable, rawData, cellHeight, cellWidths, headerRows, "Raw Input Data", "Multiple Rows Per Person");

        // Create the processed data table.
        var processedDataCoords = { x : d3.sum(cellWidths) + inputDataCoords.x + interTableSpacing, y : tableVerticalLocation };
        var processedDataTable = svg.append("g")
            .classed("table", true)
            .attr("transform", "translate(" + processedDataCoords.x + "," + processedDataCoords.y + ")");
        create_table(processedDataTable, processedDataToDisplay, cellHeight, cellWidths, headerRows, "Processed Data", "One Row Per Person");

        // Create the final data table.
        var processedTableHeight = (headerRows + inputDataRows + 1) * cellHeight;
        var finalTableHeight = (headerRows + finalNumberRows + 1) * cellHeight;
        var finalDataCoords = { x : d3.sum(cellWidths) + processedDataCoords.x + interTableSpacing,
                                y : tableVerticalLocation + (processedTableHeight / 2) - (finalTableHeight / 2) };
        var finalDataTable = svg.append("g")
            .classed("table", true)
            .attr("transform", "translate(" + finalDataCoords.x + "," + finalDataCoords.y + ")");
        create_table(finalDataTable, finalData, cellHeight, cellWidths, headerRows, "Final Data Subset", "One Row Per Person");
    });


function create_table(selection, data, cellHeight, cellWidths, headerRows, tableTitle, tableSubtitle)
{
    // Determine the number of rows and columns needed to fit the data.
    var numberRows = data.length - 1 + headerRows;
    var numberCols = data[0].length;

    // Pad the cell widths to ensure that there is one width for each column.
    while (cellWidths.length < numberCols)
    {
        cellWidths.push(cellWidths[cellWidths.length - 1]);
    }

    // Determine the width and height of the table.
    var width = d3.sum(cellWidths);
    var height = numberRows * cellHeight;

    // Create the table borders.
    selection.append("rect")
        .attr("class", "tableEdge")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height);
    selection.append("rect")
        .attr("class", "tableInternal")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height);

    // Create the rows.
    var rowY = headerRows * cellHeight;  // Skip making any row lines within the header section.
    while (rowY < height)
    {
        selection.append("line")
            .attr("class", "tableLine")
            .attr("x1", 0)
            .attr("y1", rowY)
            .attr("x2", width)
            .attr("y2", rowY);
        rowY += cellHeight;
    }

    // Create the columns.
    for (var i = 1; i < numberCols; i++)
    {
        selection.append("line")
            .attr("class", "tableLine")
            .attr("x1", d3.sum(cellWidths.slice(0, i)))
            .attr("y1", 0)
            .attr("x2", d3.sum(cellWidths.slice(0, i)))
            .attr("y2", height);
    }

    // Add text into the table.
    function text_adder(element, index)
    {
        element.forEach(function(d, i)
            {
                selection.append("text")
                    .classed("tableText", true)
                    .attr("text-anchor", "middle")
                    .attr("x", d3.sum(cellWidths.slice(0, i)) + (cellWidths[i] / 2))
                    .attr("y", (index == 0) ? (headerRows * cellHeight / 2) : ((headerRows + index - 0.5) * cellHeight))
                    .text(d);
            });
    }
    data.forEach(text_adder);

    // Add the table title and subtitle.
    selection.append("text")
        .classed("tableTitle", true)
        .attr("text-anchor", "middle")
        .attr("x", d3.sum(cellWidths) / 2)
        .attr("y", -30)
        .text(tableTitle);
    selection.append("text")
        .classed("tableSubtitle", true)
        .attr("text-anchor", "middle")
        .attr("x", d3.sum(cellWidths) / 2)
        .attr("y", -10)
        .text(tableSubtitle);
}

function generate_process_display_data(data, numberOfRecords, codesToDisplay)
{
    // Create the header.
    var header = data[0].slice(0, codesToDisplay);
    header.push("...");

    // Create the view of the data to display.
    var newData = [];
    data.forEach(function(d, i)
        {
            if (i < numberOfRecords + 1)
            {
                var datapoint = d.slice(0, codesToDisplay);
                datapoint.push("...");
                newData.push(datapoint);
            }
        });

    // Add the ... as the last row.
    var footer = ["..."];
    for (var i = 0; i < codesToDisplay; i++)
    {
        footer.push("...");
    }
    newData.push(footer);

    return newData;
}

function generate_raw_data(data, numberOfRecords)
{
    var newData = [["Patient ID", "Code"]];

    // Add a random selection of pairs of patient IDs and codes. Only select from the first quarter of the data entries.
    for (var i = 0; i < numberOfRecords; i++)
    {
        var randElement = data[Math.floor(Math.random() * data.length / 4)];
        var randCode = randElement.Codes[Math.floor(Math.random() * randElement.Codes.length)];
        newData.push([randElement.ID, randCode]);
    }

    return newData;
}

function process_data(data)
{
    // Determine which codes are in the dataset.
    var codes = new Set();
    data.forEach(function(d)
        {
            d.Codes.forEach(function(code)
                {
                    codes.add(code);
                });
        });
    codes = Array.from(codes.keys()).sort();

    // Create the header.
    var header = ["Patient ID"];
    codes.forEach(function(d) { header.push(d); });

    // Generate the processed data.
    var newData = [header];
    data.forEach(function(d)
        {
            var datapoint = [d.ID];
            codes.forEach(function(c)
                {
                    if (d.Codes.indexOf(c) > -1)
                    {
                        datapoint.push(1);
                    }
                    else
                    {
                        datapoint.push(0);
                    }
                });
            newData.push(datapoint);
        });

    return newData;
}

function subset_data(data, numberOfRecords, codesToDisplay)
{
    // Sort the datapoints by the number of codes they contain.
    var sortedData = data.slice(1).sort(function(a, b) { return d3.descending(d3.sum(a.slice(1)), d3.sum(b.slice(1))); });

    // Create a new dataset containing the numberOfRecords datapoints with the most codes.
    var newData = [data[0]];
    newData.push.apply(newData, sortedData.slice(0, numberOfRecords).sort(function(a, b) { return d3.ascending(a[0], b[0]); }));
    newData = newData.sort(function(a, b) { return d3.ascending(a.ID, b.ID); });

    // Create the view of the data to display.
    var displayData = [];
    newData.forEach(function(d, i)
        {
            if (i < numberOfRecords + 1)
            {
                var datapoint = d.slice(0, codesToDisplay);
                datapoint.push("...");
                displayData.push(datapoint);
            }
        });

    // Add the ... as the last row.
    var footer = ["..."];
    for (var i = 0; i < codesToDisplay; i++)
    {
        footer.push("...");
    }
    displayData.push(footer);

    return displayData;
}