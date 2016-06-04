var figureOffset = {top: 10, right: 10, bottom: 40, left: 40};  // The offset of the figures within the SVG element (to allow for things like axes, titles, etc.).
var svgMargin = {top: 20, right: 20, bottom: 20, left: 20};  // The margin around the whole set of figures.
var svgWidth = 600 + svgMargin.left + svgMargin.right;  // Width of the SVG element needed to hold the figure.
var svgHeight = 600 + svgMargin.top + svgMargin.bottom;  // Height of the SVG element needed to hold the figure.

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
        var rows = d3.tsv.parseRows(text);
        var data = {};
        for (i = 2; i < rows.length; i += 4)
        {
            var classLabel = rows[i][0].substring(0, rows[i][0].length - 4);
            var falsePosRates = rows[i][1].split(",").map(function(d) { return parseFloat(d); });
            var truePosRates = rows[i + 1][1].split(",").map(function(d) { return parseFloat(d); });
            var thresholds = rows[i + 2][1].split(",").map(function(d) { return parseFloat(d); });
            data[classLabel] = [];
            for (j = 0; j < falsePosRates.length; j++)
            {
                data[classLabel].push({"FPR": falsePosRates[j], "TPR": truePosRates[j], "Threshold": thresholds[j]});
            }
        }

        create_ROC_curve(svg, data);
    });

// Create a legend for the figure.
function create_legend(figureContainer, labels, legendHeight, legendWidth, legendXLoc, legendYLoc)
{
    var itemPadding = 10;  // The padding between the legend border and the items.

    // Create the legend container.
	var legend = figureContainer.append("g")
	    .classed("legend", true)
	    .attr("transform", "translate(" + legendXLoc + "," + legendYLoc + ")");

    // Create the legend items container.
    var legendItems = legend.append("g")
        .classed("legendItems", true);

    // Create the legend labels and lines.
    var labelSpacing = legendHeight / (labels.length + 1)
	for (i = 0; i < labels.length; i++)
	{
	    legendItems.append("text")
	        .attr("legendLabel", true)
            .attr("x", "5em")
            .attr("y", (i + 1) + "em")
            .text(labels[i]);
	}

	return legend
}

// Create a legend for the figure.
function create_legend2(figureContainer, labels, legendHeight, legendWidth, legendXLoc, legendYLoc)
{
    // Create the legend container.
	var legend = figureContainer.append("g")
	    .classed("legend", true)
	    .attr("transform", "translate(" + legendXLoc + "," + legendYLoc + ")");

	// Create the legend border.
	legend.append("rect")
	    .classed("legendBorder", true)
	    .attr("x", 0)
	    .attr("y", 0)
	    .attr("height", (labels.length + 1) + "em")
	    .attr("width", legendWidth);

    // Create the legend label container.
    var legendLabels = legend.append("g")
        .classed("legendItems", true);

    // Create the legend labels.
    var labelSpacing = legendHeight / (labels.length + 1)
	for (i = 0; i < labels.length; i++)
	{
	    legendLabels.append("text")
	        .attr("legendLabel", true)
            //.attr("text-anchor", "middle")
            .attr("x", "5em")
            .attr("y", (i + 1) + "em")
            .text(labels[i]);
	}

	return legend
}

// Generate the ROC curves for all classes.
function create_ROC_curve(figureContainer, data)
{
    // Create scales for the figure.
    var xScale = d3.scale.linear()
        .domain([0.0, 1.0])
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
        .text("False Positive Rate");
    yAxisLabelLoc = { x : 0, y : (svgHeight - figureOffset.bottom) / 2 };
    var yAxisLabel = svg.append("text")
        .classed("label", true)
        .attr("text-anchor", "middle")
        .attr("x", yAxisLabelLoc.x)
        .attr("y", yAxisLabelLoc.y)
        .attr("transform", "rotate(-90, " + yAxisLabelLoc.x + ", " + yAxisLabelLoc.y + ")")
        .text("True Positive Rate");

    // Add the title to the figure.
    var title = figureContainer.append("text")
        .classed("title", true)
        .attr("text-anchor", "middle")
        .attr("x", (svgWidth + figureOffset.left) / 2)
        .attr("y", 0)
        .text("Training Set ROC Curves");

    // Add the dashed line between (0, 0) and (1, 1).
    svg.append("line")
        .attr("x1", xScale(0))
        .attr("y1", yScale(0))
        .attr("x2", xScale(1))
        .attr("y2", yScale(1))
        .attr("stroke-dasharray", "10,10")
        .style("stroke", "black");

    // Create the line generator.
    var line = d3.svg.line()
        .x(function(d) { return xScale(d.FPR); })
        .y(function(d) { return yScale(d.TPR); })
        .interpolate("basis");

	// Add datapoints.
	for (var i in data)
	{
	    var rocCurve = figureContainer.append("path")
	        .datum(data[i])
	        .classed("ROC", true)
	        .classed(i.replace(/ /g, "_"), true)
			.attr("d", line);
	}

	// Create the legend.
	var legend = create_legend(figureContainer, Object.keys(data), 100, 200, xScale(0.7), yScale(0.3));
}