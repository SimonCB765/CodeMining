var figureWidth = 400;  // Height of the entire figure for an experiment's results including labels and title.
var figureHeight = 400;  // Width of the entire figure for an experiment's results including labels and title.
var figureMargin = {top: 10, right: 10, bottom: 10, left: 10};  // Margin around each individual figure.
var figureOffset = {top: 10, right: 10, bottom: 30, left: 30};  // The offset of the figures within the SVG element (to allow for things like axes, titles, etc.).
var svgWidth = (2 * (figureWidth + figureMargin.left + figureMargin.right));  // Width of the SVG element needed to hold both figures.
var svgHeight = figureHeight + figureMargin.top + figureMargin.bottom;  // Height of the SVG element needed to hold both figures.
var svgMargin = {top: 10, right: 10, bottom: 10, left: 20};  // The margin around the whole set of figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth + svgMargin.left + svgMargin.right)
        .attr("height", svgHeight + svgMargin.top + svgMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + svgMargin.left + ", " + svgMargin.top + ")");

// Add the axes labels. To rotate the label on the y axis we use a transformation. As this will alter the coordinate system being used
// to place the text, we perform the rotation around the point where the text is centered. This causes the text to rotate but not move.
var xAxisLabel = svg.append("text")
    .attr("class", "label")
    .attr("text-anchor", "middle")
    .attr("x", svgWidth / 2)
    .attr("y", svgHeight)
    .text("False Positive Rate");
yAxisLabelLoc = { x : 0, y : svgHeight / 2 };
var yAxisLabel = svg.append("text")
    .attr("class", "label")
    .attr("text-anchor", "middle")
    .attr("x", yAxisLabelLoc.x)
    .attr("y", yAxisLabelLoc.y)
    .attr("transform", "rotate(-90, " + yAxisLabelLoc.x + ", " + yAxisLabelLoc.y + ")")
    .text("True Positive Rate");

// Read the data in.
var dataAccessorFunction = function(d)
    {
		delete d["ClassificationThreshold"];  // Don't care about the fold number column.
		Object.keys(d).forEach(function(key)
			{
				// Convert the entry from a comma separated string into an object holding the 4 values separately.
				var value = d[key].split(',');
				value = {"TP" : +value[0], "FP" : +value[1], "TN" : +value[2], "FN" : +value[3]};
				d[key] = value;
			});
        return d;
    }
d3.tsv("/Data/T1VT2_InitialModelCVResults.tsv", dataAccessorFunction, function(error, initialData)
    {
		initialResults = aggregate_results(initialData);
		initialMetrics = calculate_metrics(initialResults);
		
		d3.tsv("/Data/T1VT2_FinalModelCVResults.tsv", dataAccessorFunction, function(error, finalData)
			{
				finalResults = aggregate_results(finalData);
				finalMetrics = calculate_metrics(finalResults);
				
				// Create the figure with the two curves.
				var figureContainer = svg.append("g")
                    .attr("transform", "translate(" + figureMargin.left + ", " + figureMargin.top + ")");
				create_ROC_curve(figureContainer, initialMetrics, finalMetrics, "Placeholder Title");
			}
		);
    }
);
d3.tsv("/Data/DVND_InitialModelCVResults.tsv", dataAccessorFunction, function(error, initialData)
    {
		initialResults = aggregate_results(initialData);
		initialMetrics = calculate_metrics(initialResults);
		
		d3.tsv("/Data/DVND_FinalModelCVResults.tsv", dataAccessorFunction, function(error, finalData)
			{
				finalResults = aggregate_results(finalData);
				finalMetrics = calculate_metrics(finalResults);
				
				// Create the figure with the two curves.
				var xPosition = figureMargin.left + figureWidth + figureMargin.right + figureMargin.left;
				var figureContainer = svg.append("g")
					.attr("transform", "translate(" + xPosition + ", " + figureMargin.top + ")");
				create_ROC_curve(figureContainer, initialMetrics, finalMetrics, "Placeholder Title");
			}
		);
    }
);

// Combine the results across thresholds, and collect them all in one object with an entry for each threshold.
function aggregate_results(data)
{
	// Create an object to hold the aggregate results from all the CV folds.
	var thresholdsUsed = Object.keys(data[0]);  // Determine the thresholds at which performance was calculated.
	var results = thresholdsUsed.reduce(function(objectBeingCreated, currentValue)
		{
			objectBeingCreated[currentValue] = {"TP" : 0, "FP" : 0, "TN" : 0, "FN" : 0};
			return objectBeingCreated;
		}, {});
	
	// Fill the result object.
	data.forEach(function(row)
		{
			thresholdsUsed.forEach(function(threshold)
				{
					var thresholdResult = row[threshold];
					results[threshold]["TP"] += thresholdResult["TP"];
					results[threshold]["FP"] += thresholdResult["FP"];
					results[threshold]["TN"] += thresholdResult["TN"];
					results[threshold]["FN"] += thresholdResult["FN"];
				});
		});

	return results;
}

// Calculate the metrics needed for producing the ROC curve (true and false positive rates).
function calculate_metrics(results)
{
	var metrics = {};
	Object.keys(results).forEach(function(key)
		{
			var thresholdResult = results[key];
			var sensitivity = thresholdResult["TP"] / (thresholdResult["TP"] + thresholdResult["FN"]);
			var specificity = thresholdResult["TN"] / (thresholdResult["TN"] + thresholdResult["FP"]);
			metrics[key] = {"TPR" : sensitivity, "FPR" : (1 - specificity)};
		});
	metrics = Object.keys(metrics).map(function(x) { return metrics[x] });  // Convert metrics to an array of objects.
	metrics.sort(function(a, b)
		{
			return (a.FPR < b.FPR) ? -1 : 1;
		});
	return metrics;
}

function create_ROC_curve(figureContainer, initialMetrics, finalMetrics, figureTitle)
{
    // Create scales for the figure.
    var xScale = d3.scale.linear()
        .domain([0.0, 1.0])
        .range([figureOffset.left, figureWidth - figureOffset.right]);
    var yScale = d3.scale.linear()
        .domain([0.0, 1.0])
        .range([figureHeight - figureOffset.bottom, figureOffset.top]);

    // Add the axes for the figure.
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    xAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (figureHeight - figureOffset.bottom) + ")")
        .call(xAxis);
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");
    yAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(" + figureOffset.left + ", 0)")
        .call(yAxis);

    // Add the title to the figure.
    var title = figureContainer.append("text")
        .attr("class", "title")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figureOffset.left - figureOffset.right) / 2) + figureOffset.left)
        .attr("y", figureOffset.top * 3 / 4)
        .text(figureTitle);

	// Add datapoint.
	console.log(initialMetrics);
	var initialDatapoints = figureContainer.selectAll(".initial")
		.data(initialMetrics)
		.enter()
			.append("circle")
				.classed("initial", true)
				.attr("cx", function(d) { return xScale(d.FPR); })
				.attr("cy", function(d) { return yScale(d.TPR); })
				.attr("r", 2);
	var finalDatapoints = figureContainer.selectAll(".final")
		.data(finalMetrics)
		.enter()
			.append("circle")
				.classed("final", true)
				.attr("cx", function(d) { return xScale(d.FPR); })
				.attr("cy", function(d) { return yScale(d.TPR); })
				.attr("r", 2);
}