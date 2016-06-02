var figureWidth = 400;  // Height of the entire figure including labels and title.
var figureHeight = 400;  // Width of the entire figure including labels and title.
var figureMargin = {top: 10, right: 10, bottom: 10, left: 10};  // Margin around each individual figure.
var svgWidth = (figureWidth + figureMargin.left + figureMargin.right) * 2;  // Width of the SVG element needed to hold both figures and their padding.
var svgHeight = figureHeight + figureMargin.top + figureMargin.bottom;  // Height of the SVG element needed to hold both figures and their padding.
var svgMargin = {top: 10, right: 10, bottom: 10, left: 10};  // The margin around the set of figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth + svgMargin.left + svgMargin.right)
        .attr("height", svgHeight + svgMargin.top + svgMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + svgMargin.left + ", " + svgMargin.top + ")");

// Read the data in.
d3.text(dataset, function(text)
    {
        var data = [];
        var rows = text.split("\n");  // Rows of the file.
        var header = rows[0].split("\t");  // The file header.
        var numberClasses = (header.length - 3) / 2;  // Two columns per class, and three unrelated to class.
        var chosenClass = header[2].match(/_.*_/)[0];  // The class that will have the posteriors plotted.
        chosenClass = chosenClass.substring(1, chosenClass.length - 1);
        for (i = 1; i < rows.length - 1; i++)
        {
            var chunks = rows[i].split("\t");
            var datum = {};
            datum["PatientID"] = parseInt(chunks[0]);
            datum["Initial"] = parseFloat(chunks[2]);
            datum["Final"] = parseFloat(chunks[2 + 1 + numberClasses]);
            data.push(datum);
        }

        // Create the figure for the line graph.
        var lineGraphContainer = svg.append("g").attr("transform", "translate(" + figureMargin.left + ", " + figureMargin.top + ")");
        createLineGraph(lineGraphContainer, data, "Placeholder Title");

        // Create the figure for the histogram.
        var xPosHisto = figureMargin.left + figureWidth + figureMargin.right + figureMargin.left;
        var histogramContainer = svg.append("g").attr("transform", "translate(" + xPosHisto + ", " + figureMargin.top + ")");
        createHistogram(histogramContainer, data, "Placeholder Title");
    }
);

function createHistogram(figureContainer, dataArray, figureTitle)
{
    // Define positioning variables.
    var figurePadding = {top: 40, right: 5, bottom: 50, left: 60};  // Padding around the graph in the figure to allow for titles and labelling axes.
    var binOffset = 4;  // The offset for the bins so that each set of bins is more clearly separated from the others.

    // Bin the data.
    var numberOfBins = 10;
    var modelOneBinned = d3.layout.histogram()
        .value(function(d) { return d.Initial; })
        .bins(numberOfBins)
        .frequency(true)
        .range([0, 1])
        (dataArray);
    var maxCountFirstModel = d3.max(modelOneBinned, function(d) { return d.y; });
    var modelTwoBinned = d3.layout.histogram()
        .value(function(d) { return d.Final; })
        .bins(numberOfBins)
        .frequency(true)
        .range([0, 1])
        (dataArray);
    var maxCountSecondModel = d3.max(modelTwoBinned, function(d) { return d.y; });

    // Create scales for the figure.
    var maxCount = Math.ceil(Math.max(maxCountFirstModel, maxCountSecondModel) * 0.1) / 0.1;  // Maximum count rounded up to nearest 10.
    var xScale = d3.scale.linear()
        .domain([0, 1.0])
        .range([figurePadding.left, figureWidth - figurePadding.right]);
    var yScale = d3.scale.linear()
        .domain([0, maxCount])
        .range([figureHeight - figurePadding.bottom, figurePadding.top]);

    // Add the axes for the figure.
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    xAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (figureHeight - figurePadding.bottom) + ")")
        .call(xAxis);
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");
    yAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(" + figurePadding.left + ", 0)")
        .call(yAxis);

    // Add the title to the figure.
    var title = figureContainer.append("text")
        .attr("class", "title")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figurePadding.top * 3 / 4)
        .text(figureTitle);

    // Add the labels on the axes. To rotate the label on the y axis we use a transformation. As this will alter the coordinate system being used
    // to place the text, we perform the rotation around the point where the text is centered. This causes the text to rotate but not move.
    var xAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figureHeight - (figurePadding.bottom / 4))
        .text("Posterior Probability");
    yAxisLabelLoc = { x : figurePadding.left / 3, y : ((figureHeight - figurePadding.top - figurePadding.bottom) / 2) + figurePadding.top };
    var yAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", yAxisLabelLoc.x)
        .attr("y", yAxisLabelLoc.y)
        .attr("transform", "rotate(-90, " + yAxisLabelLoc.x + ", " + yAxisLabelLoc.y + ")")
        .text("Number of Examples");

    // Add the histogram for the initial model.
    var modelOneBars = figureContainer.selectAll(".modelOneBar")
        .data(modelOneBinned)
        .enter()
            .append("g")
                .classed("firstModel bar", true)
                .attr("transform", function(d) { return "translate(" + (xScale(d.x) + binOffset) + ", " + yScale(d.y) + ")"; });
    modelOneBars.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        //.attr("width", function(d) { return xScale(d.x + d.dx) - xScale(d.x) - 1; })
        .attr("width", function(d) { return ((xScale(d.x + d.dx) - xScale(d.x)) / 2) - binOffset; })
        .attr("height", function(d) { return yScale(0) - yScale(d.y); });

    // Add the histogram for the final model.
    var modelTwoBars = figureContainer.selectAll(".modelTwoBar")
        .data(modelTwoBinned)
        .enter()
            .append("g")
                .classed("secondModel bar", true)
                //.attr("transform", function(d) { return "translate(" + xScale(d.x) + ", " + yScale(d.y) + ")"; });
                .attr("transform", function(d) { return "translate(" + xScale(d.x + (d.dx / 2)) + ", " + yScale(d.y) + ")"; });
    modelTwoBars.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        //.attr("width", function(d) { return xScale(d.x + d.dx) - xScale(d.x) - 1; })
        .attr("width", function(d) { return ((xScale(d.x + d.dx) - xScale(d.x)) / 2) - binOffset; })
        .attr("height", function(d) { return yScale(0) - yScale(d.y); });
}

function createLineGraph(figureContainer, dataArray, figureTitle)
{
    // Define positioning variables.
    var figurePadding = {top: 40, right: 5, bottom: 50, left: 60};  // Padding around the graph in the figure to allow for titles and labelling axes.

    // Create scales for the figure.
    var xScale = d3.scale.linear()
        .domain([0, dataArray.length + 1])  // Add 1 to allow for the final datapoint to have space for a line.
        .range([figurePadding.left, figureWidth - figurePadding.right]);
    var yScale = d3.scale.linear()
        .domain([0.0, 1.0])
        .range([figureHeight - figurePadding.bottom, figurePadding.top]);

    // Add the axes for the figure.
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    xAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (figureHeight - figurePadding.bottom) + ")")
        .call(xAxis);
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");
    yAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(" + figurePadding.left + ", 0)")
        .call(yAxis);

    // Add the title to the figure.
    var title = figureContainer.append("text")
        .attr("class", "title")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figurePadding.top * 3 / 4)
        .text(figureTitle);

    // Add the labels on the axes. To rotate the label on the y axis we use a transformation. As this will alter the coordinate system being used
    // to place the text, we perform the rotation around the point where the text is centered. This causes the text to rotate but not move.
    var xAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figureHeight - (figurePadding.bottom / 4))
        .text("Example Index");
    yAxisLabelLoc = { x : figurePadding.left / 3, y : ((figureHeight - figurePadding.top - figurePadding.bottom) / 2) + figurePadding.top };
    var yAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", yAxisLabelLoc.x)
        .attr("y", yAxisLabelLoc.y)
        .attr("transform", "rotate(-90, " + yAxisLabelLoc.x + ", " + yAxisLabelLoc.y + ")")
        .text("Posterior Probability");

    // Add the line for the initial model after sorting the data in ascending order of the initial model posterior.
    dataArray = dataArray.sort(function(a, b) { return d3.ascending(a.Initial, b.Initial); });
    var dataPathFirstModel = "M" + xScale(0) + "," + yScale(0);
    dataArray.forEach(function(d, index)
    {
        dataPathFirstModel += "L" + xScale(index) + "," + yScale(d.Initial) + "h1";
    });
    var dataLineFirstModel = figureContainer.append("path")
        .classed("firstModel line", true)
        .attr("d", dataPathFirstModel);

    // Add the line for the final model after sorting the data in ascending order of the final model posterior.
    dataArray = dataArray.sort(function(a, b) { return d3.ascending(a.Final, b.Final); });
    var dataPathSecondModel = "M" + xScale(0) + "," + yScale(0);
    dataArray.forEach(function(d, index)
    {
        dataPathSecondModel += "L" + xScale(index) + "," + yScale(d.Final) + "h1";
    });
    var dataLineSecondModel = figureContainer.append("path")
        .classed("secondModel line", true)
        .attr("d", dataPathSecondModel);
}