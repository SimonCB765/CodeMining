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
        var numberClasses = (header.length - 4) / 2;  // Two columns per class, and four unrelated to class.
        var classesUsed = new Set();
        for (i = 1; i < rows.length - 1; i++)
        {
            var chunks = rows[i].split("\t");
            var datum = {};
            datum["PatientID"] = parseInt(chunks[0]);
            datum["Class"] = chunks[1];
            classesUsed.add(chunks[1]);
            for (j = 4; j < 4 + numberClasses; j++)
            {
                var classLabel = header[j].match(/_.*/)[0].substring(1);
                datum[classLabel] = parseFloat(chunks[j]);
            }
            data.push(datum);
        }
        console.log(datum);
        classesUsed = Array.from(classesUsed);

        // Create the left discard graph.
        var discardContainer = svg.append("g").attr("transform", "translate(" + figureMargin.left + ", " + figureMargin.top + ")");
        create_discard_graph(discardContainer, data, classesUsed[0], "PLACEHOLDER TITLE", cutoff);

        // Create the right discard graph.
        var xPosRightGraph = figureMargin.left + figureWidth + figureMargin.right + figureMargin.left;
        discardContainer = svg.append("g").attr("transform", "translate(" + xPosRightGraph + ", " + figureMargin.top + ")");
        create_discard_graph(discardContainer, data, classesUsed[1], "PLACEHOLDER TITLE", cutoff);
    }
);

function create_discard_graph(figureContainer, dataArray, classToPlot, figureTitle, cutoff)
{
    // Filter out all examples not in the class of interest.
    dataArray = dataArray.filter(function(d) { return (d["Class"] === classToPlot) ? true : false; });

    // Sort in ascending order of posterior of the class of interest.
    dataArray = dataArray.sort(function(a, b) { return d3.ascending(a[classToPlot], b[classToPlot]); });

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

    // Add the line for the class of interest.
    var dataPath = "M" + xScale(0) + "," + yScale(0);
    dataArray.forEach(function(d, index)
    {
        dataPath += "L" + xScale(index) + "," + yScale(d[classToPlot]) + "h1";
    });
    var dataLine = figureContainer.append("path")
        .classed(classToPlot + " line", true)
        .attr("d", dataPath);
}