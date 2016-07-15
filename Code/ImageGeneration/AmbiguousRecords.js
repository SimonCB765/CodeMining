var figureOffset = {top: 10, right: 10, bottom: 40, left: 40};  // The offset of the figures within the SVG element (to allow for things like axes, titles, etc.).
var svgMargin = {top: 20, right: 20, bottom: 20, left: 20};  // The margin around the whole set of figures.
var svgWidth = 400 + svgMargin.left + svgMargin.right;  // Width of the SVG element needed to hold the figure.
var svgHeight = 400 + svgMargin.top + svgMargin.bottom;  // Height of the SVG element needed to hold the figure.

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

        create_line_graph(svg, data);
    });

function create_line_graph(figureContainer, dataArray)
{
    // Create scales for the figure.
    var xScale = d3.scale.linear()
        .domain([0, dataArray.length + 1])  // Add 1 to allow for the final datapoint to have space for a line.
        .range([figureOffset.left, svgWidth - figureOffset.right]);
    var yScale = d3.scale.linear()
        .domain([0.0, 1.0])
        .range([svgHeight - figureOffset.bottom, figureOffset.top]);

    // Add the axes for the figure.
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    xAxis = figureContainer.append("g")
        .classed("axis", true)
        .attr("transform", "translate(0, " + (svgHeight - figureOffset.bottom) + ")")
        .call(xAxis);
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");
    yAxis = figureContainer.append("g")
        .classed("axis", true)
        .attr("transform", "translate(" + figureOffset.left + ", 0)")
        .call(yAxis);

    // Add the axes labels. To rotate the label on the y axis we use a transformation. As this will alter the coordinate system being used
    // to place the text, we perform the rotation around the point where the text is centered. This causes the text to rotate but not move.
    var xAxisLabel = svg.append("text")
        .classed("label", true)
        .attr("text-anchor", "middle")
        .attr("x", (svgWidth + figureOffset.left) / 2)
        .attr("y", svgHeight)
        .text("Example Index");
    yAxisLabelLoc = { x : 0, y : (svgHeight - figureOffset.bottom) / 2 };
    var yAxisLabel = svg.append("text")
        .classed("label", true)
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
        if (index % 2 === 0)
        {
            dataPathFirstModel += "L" + xScale(index) + "," + yScale(d.Initial) + "h1";
        }
    });
    var dataLineFirstModel = figureContainer.append("path")
        .classed("Initial_Model line", true)
        .attr("d", dataPathFirstModel);

    // Add the line for the final model after sorting the data in ascending order of the final model posterior.
    dataArray = dataArray.sort(function(a, b) { return d3.ascending(a.Final, b.Final); });
    var dataPathSecondModel = "M" + xScale(0) + "," + yScale(0);
    dataArray.forEach(function(d, index)
    {
        dataPathSecondModel += "L" + xScale(index) + "," + yScale(d.Final) + "h1";
    });
    var dataLineSecondModel = figureContainer.append("path")
        .classed("Final_Model line", true)
        .attr("d", dataPathSecondModel);

    // Add the legend.
    var legend = create_path_legend(figureContainer, ["Initial Model", "Final Model"], xScale(dataArray.length / 10), yScale(0.75))
}